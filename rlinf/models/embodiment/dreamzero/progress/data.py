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

"""Training data for the DreamZero progress value model (design §8, §9).

Two real LIBERO sources, read directly with their own pixels and outcome labels
(design §8.1 -- NVIDIA outcomes stay attached to NVIDIA frames; action hashes
are used only for grouped splitting, never to transfer a label):

- **PI** ``physical-intelligence/libero`` -- LeRobot v2 parquet, 1,693 episodes,
  all successes. Exterior ``image`` / wrist ``wrist_image``, **no mirror**.
- **NVIDIA** ``nvidia/LIBERO-Cosmos-Policy/all_episodes`` -- per-episode HDF5,
  2,000 episodes (1,700 success + 300 failure). Exterior
  ``primary_images_jpeg`` / wrist ``wrist_images_jpeg``, **horizontally
  mirrored** to match the PI orientation (design §8.2/§8.3). Both outcomes are
  used: successes feed progress regression + TD; failures feed TD only.

Each training sample is an **adjacent-frame transition** ``(o_t, o_{t+1})`` with
the language instruction and the temporal index ``t`` / episode length ``T``
(for the ``t/T`` progress label and the windowed TD reward, both derived in
:mod:`...progress.losses`, not stored per sample). Frames are decoded lazily per
access -- each episode is opened once and cached -- so only lightweight
per-episode references live in memory.

Splitting is grouped by ``(LIBERO task, SHA256 of the float32 action sequence)``
so frames from one episode -- and PI/NVIDIA episodes sharing an action sequence
-- never straddle the train/val/test boundary (design §8.4).

:class:`SmolVLAProgressCollator` then makes the training pixels match the dreams
the model scores at inference, in two stages: first it puts each frame through
the **frozen IDM/SFT video geometry** (``VideoCrop(0.95)`` -> resize ->
exterior|wrist concat -> squash to the WAM 160x320 canvas -> split into two
160x160 views), so training shares the dreams' field-of-view and resolution
rather than a raw-256 resize (data README §4.1); then it applies the **official
SmolVLA preprocessing** (resize-with-pad to the SigLIP size, ``[-1, 1]``
normalization, verbatim-instruction tokenization).
"""

import hashlib
import io
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from rlinf.utils.logging import get_logger

logger = get_logger()

PI_EXTERIOR_KEY = "image"
PI_WRIST_KEY = "wrist_image"
NVIDIA_EXTERIOR_KEY = "primary_images_jpeg"
NVIDIA_WRIST_KEY = "wrist_images_jpeg"
ACTION_COLUMN_ALIASES = ("actions", "action")

SOURCE_PI = "pi"
SOURCE_NVIDIA = "nvidia"

# Fixed DreamZero geometry constants (not tunable -- the model always matches
# the WAM dream canvas). The TD reward / terminal window live in losses.py.
DREAM_VIEW_SIZE = 160  # WAM canvas is 160x320 -> two 160x160 views
CROP_SCALE = 0.95  # frozen IDM/SFT VideoCrop scale

# Per-episode image cache: open an episode once, serve all its frames from RAM
# (matches IDM's ``_read_v2_episode`` table cache). One cache per worker process.
EPISODE_CACHE_SIZE = 50


# ----------------------------------------------------------------------
# Decoding / hashing helpers
# ----------------------------------------------------------------------


def _decode_image_bytes(raw: Any, mirror: bool) -> np.ndarray:
    """Decode PNG/JPEG bytes (or an HF Image struct) to ``[H, W, 3]`` uint8 RGB.

    ``mirror`` applies the horizontal flip NVIDIA frames need (design §8.2).
    """
    from PIL import Image, ImageOps

    if isinstance(raw, dict):  # HF Image feature: {"bytes": ..., "path": ...}
        raw = raw.get("bytes") if raw.get("bytes") is not None else raw.get("path")
    if isinstance(raw, (bytes, bytearray)):
        img = Image.open(io.BytesIO(raw))
    elif isinstance(raw, str):  # path on disk
        img = Image.open(raw)
    elif isinstance(raw, np.ndarray):  # already-decoded array
        img = Image.fromarray(raw)
    else:
        raise TypeError(f"Unsupported image cell type: {type(raw)}")
    img = img.convert("RGB")
    if mirror:
        img = ImageOps.mirror(img)
    return np.asarray(img, dtype=np.uint8)


def action_sha256(actions: np.ndarray) -> str:
    """SHA256 of the float32 action sequence for provenance / grouped splitting.

    Casting to float32 first makes PI (float32) and NVIDIA (float64) episodes
    that share an action sequence hash identically (failure-data README §2).
    """
    arr = np.ascontiguousarray(np.asarray(actions, dtype=np.float32))
    return hashlib.sha256(arr.tobytes()).hexdigest()


# ----------------------------------------------------------------------
# Episode references (lazy frame access)
# ----------------------------------------------------------------------


@dataclass
class ProgressEpisode:
    """Lightweight reference to one episode; frames are decoded on demand.

    ``num_frames`` is the trajectory length; :meth:`load_views` returns the
    exterior/wrist uint8 frames for a single timestep.
    """

    source: str
    task: str
    success: bool
    num_frames: int
    action_hash: str
    loader: Callable[[int], tuple[np.ndarray, np.ndarray]]
    group_key: tuple = field(default=())

    def load_views(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        return self.loader(t)


# ----------------------------------------------------------------------
# PI LeRobot v2 reader
# ----------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pi_episode_language(meta_dir: Path) -> dict[int, str]:
    """Map ``episode_index -> instruction`` from ``meta/episodes.jsonl``.

    ``tasks[0]`` is the verbatim instruction; if it is a task-index int, resolve
    it through ``meta/tasks.jsonl`` (design data README §1).
    """
    tasks_lookup: dict[int, str] = {}
    tasks_path = meta_dir / "tasks.jsonl"
    if tasks_path.exists():
        with open(tasks_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    tasks_lookup[int(rec["task_index"])] = rec["task"]

    out: dict[int, str] = {}
    with open(meta_dir / "episodes.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            ep = int(rec["episode_index"])
            tasks = rec.get("tasks") or []
            if not tasks:
                continue
            first = tasks[0]
            out[ep] = tasks_lookup.get(int(first), str(first)) if isinstance(
                first, int
            ) else str(first)
    return out


def _resolve_action_column(schema_names: list[str]) -> str:
    for name in ACTION_COLUMN_ALIASES:
        if name in schema_names:
            return name
    raise KeyError(f"No action column in {schema_names}; tried {ACTION_COLUMN_ALIASES}")


@lru_cache(maxsize=EPISODE_CACHE_SIZE)
def _read_pi_image_columns(parquet_path: str) -> tuple[list, list]:
    """Cached read of one PI episode's image columns (IDM-style episode cache).

    Keeps frame bytes out of the resident episode list: enumeration reads only
    the action column, and frames are pulled per episode on demand. The two
    reads in one transition (``t`` and ``t+1``) share a hit, and up to
    :data:`EPISODE_CACHE_SIZE` recent episodes stay resident for cross-sample
    reuse.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=[PI_EXTERIOR_KEY, PI_WRIST_KEY])
    return (
        table.column(PI_EXTERIOR_KEY).to_pylist(),
        table.column(PI_WRIST_KEY).to_pylist(),
    )


def read_pi_episodes(root: str) -> list[ProgressEpisode]:
    """Enumerate PI LeRobot v2 episodes (all successes) as lazy references."""
    import pyarrow.parquet as pq

    root = Path(root)
    meta_dir = root / "meta"
    info = _load_json(meta_dir / "info.json")
    languages = _pi_episode_language(meta_dir)
    chunks_size = int(info.get("chunks_size") or 1000)
    data_tmpl = info["data_path"]

    episodes: list[ProgressEpisode] = []
    for ep in sorted(languages):
        rel = data_tmpl.format(episode_chunk=ep // chunks_size, episode_index=ep)
        parquet_path = str(root / rel)
        schema_names = list(pq.read_schema(parquet_path).names)
        action_col = _resolve_action_column(schema_names)
        actions = np.asarray(
            pq.read_table(parquet_path, columns=[action_col]).column(action_col).to_pylist(),
            dtype=np.float32,
        )
        task = languages[ep]
        action_hash = action_sha256(actions)

        def _loader(t, _pp=parquet_path):
            ext, wri = _read_pi_image_columns(_pp)
            return (
                _decode_image_bytes(ext[t], mirror=False),
                _decode_image_bytes(wri[t], mirror=False),
            )

        episodes.append(
            ProgressEpisode(
                source=SOURCE_PI,
                task=task,
                success=True,
                num_frames=int(actions.shape[0]),
                action_hash=action_hash,
                loader=_loader,
                group_key=(task, action_hash),
            )
        )
    logger.info("PI: %d episodes from %s", len(episodes), root)
    return episodes


# ----------------------------------------------------------------------
# NVIDIA Cosmos HDF5 reader (success + failure)
# ----------------------------------------------------------------------


def _nvidia_success_from_name(name: str) -> Optional[bool]:
    for part in name.split("--"):
        if part.startswith("success="):
            return part.split("=", 1)[1].lower().startswith("t")
    return None


@lru_cache(maxsize=EPISODE_CACHE_SIZE)
def _read_nvidia_image_datasets(path: str) -> tuple[list, list]:
    """Cached read of one NVIDIA episode's JPEG image datasets (IDM-style).

    Opens the HDF5 file **once** and returns the full exterior/wrist encoded-byte
    rows, so per-frame access serves from RAM instead of reopening the file each
    time (the PI reader and IDM's ``_read_v2_episode`` cache the same way). Bytes
    stay encoded; decode + mirror happen lazily in :func:`_decode_image_bytes`.
    """
    import h5py

    with h5py.File(path, "r") as fh:
        ext = list(fh[NVIDIA_EXTERIOR_KEY][:])
        wri = list(fh[NVIDIA_WRIST_KEY][:])
    return ext, wri


def read_nvidia_episodes(root: str) -> list[ProgressEpisode]:
    """Enumerate NVIDIA Cosmos HDF5 episodes (success **and** failure).

    Frames are JPEG-decoded and **horizontally mirrored** (design §8.2). The
    episode outcome is read from the ``success`` HDF5 attribute (falling back to
    the filename), and both outcomes are kept.
    """
    import h5py

    root = Path(root)
    files = sorted(root.glob("*.hdf5")) + sorted(root.glob("**/*.hdf5"))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No .hdf5 episodes under {root}")

    episodes: list[ProgressEpisode] = []
    for path in files:
        with h5py.File(path, "r") as f:
            success_attr = f.attrs.get("success")
            success = (
                bool(success_attr)
                if success_attr is not None
                else _nvidia_success_from_name(path.name)
            )
            if success is None:
                logger.warning("NVIDIA episode without success label, skipping: %s", path)
                continue
            task_attr = f.attrs.get("task_description", "")
            task = (
                task_attr.decode() if isinstance(task_attr, bytes) else str(task_attr)
            )
            num_frames = int(f[NVIDIA_EXTERIOR_KEY].shape[0])
            actions = np.asarray(f["actions"][:], dtype=np.float32)
        action_hash = action_sha256(actions)

        def _loader(t, _path=str(path)):
            ext_rows, wri_rows = _read_nvidia_image_datasets(_path)
            ext, wri = ext_rows[t], wri_rows[t]
            ext = ext.tobytes() if isinstance(ext, np.ndarray) else bytes(ext)
            wri = wri.tobytes() if isinstance(wri, np.ndarray) else bytes(wri)
            return (
                _decode_image_bytes(ext, mirror=True),
                _decode_image_bytes(wri, mirror=True),
            )

        episodes.append(
            ProgressEpisode(
                source=SOURCE_NVIDIA,
                task=task,
                success=bool(success),
                num_frames=num_frames,
                action_hash=action_hash,
                loader=_loader,
                group_key=(task, action_hash),
            )
        )
    n_succ = sum(e.success for e in episodes)
    logger.info(
        "NVIDIA: %d episodes from %s (%d success / %d failure)",
        len(episodes),
        root,
        n_succ,
        len(episodes) - n_succ,
    )
    return episodes


# ----------------------------------------------------------------------
# Grouped splitting (design §8.4)
# ----------------------------------------------------------------------


def grouped_split(
    episodes: list[ProgressEpisode],
    val_fraction: float = 0.05,
    test_fraction: float = 0.05,
    seed: int = 0,
) -> dict[str, list[ProgressEpisode]]:
    """Split episodes into train/val/test grouped by ``(task, action_hash)``.

    Whole groups go to one split, so an episode -- and any PI/NVIDIA pair that
    shares an action sequence -- never leaks across the boundary.
    """
    groups: dict[tuple, list[ProgressEpisode]] = {}
    for ep in episodes:
        groups.setdefault(ep.group_key, []).append(ep)
    keys = sorted(groups)
    order = np.random.default_rng(seed).permutation(len(keys))
    n = len(keys)
    n_test = max(1, int(n * test_fraction))
    n_val = max(1, int(n * val_fraction))
    test_keys = {keys[i] for i in order[:n_test]}
    val_keys = {keys[i] for i in order[n_test : n_test + n_val]}

    out = {"train": [], "val": [], "test": []}
    for key in keys:
        split = "test" if key in test_keys else "val" if key in val_keys else "train"
        out[split].extend(groups[key])
    logger.info(
        "grouped_split: %d groups -> train=%d val=%d test=%d episodes",
        n,
        len(out["train"]),
        len(out["val"]),
        len(out["test"]),
    )
    return out


# ----------------------------------------------------------------------
# Transition dataset
# ----------------------------------------------------------------------


class ProgressTransitionDataset(Dataset):
    """Adjacent-frame transitions ``(o_t, o_{t+1})`` over the combined sources.

    One item per frame ``t`` of every episode. The final frame (``t == T-1``)
    has no successor, so its ``next`` views repeat ``o_t``; this is harmless
    because that frame falls inside the terminal window, where the TD target
    does not bootstrap (``done == 1``), so the repeat is never used.

    Returns raw uint8 views, the language string, and ``(t_index, episode_len,
    success)``; the windowed TD reward / ``done`` (terminal ``+1`` success /
    ``-1`` failure, ``0`` elsewhere) are derived from these in
    :func:`...progress.losses.td_loss`, not stored per sample. SmolVLA
    preprocessing / tokenization is deferred to :class:`SmolVLAProgressCollator`.
    """

    def __init__(self, episodes: list[ProgressEpisode]):
        self.episodes = episodes
        # Flat (episode_idx, frame_t) index over all usable frames.
        self.index: list[tuple[int, int]] = []
        for ei, ep in enumerate(episodes):
            self.index.extend((ei, t) for t in range(ep.num_frames))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, Any]:
        ei, t = self.index[i]
        ep = self.episodes[ei]
        ext_t, wri_t = ep.load_views(t)
        if t == ep.num_frames - 1:
            ext_n, wri_n = ext_t, wri_t  # no successor; unused (terminal window)
        else:
            ext_n, wri_n = ep.load_views(t + 1)
        return {
            "exterior": ext_t,
            "wrist": wri_t,
            "next_exterior": ext_n,
            "next_wrist": wri_n,
            "language": ep.task,
            "t_index": t,
            "episode_len": ep.num_frames,
            "success": ep.success,
            "source": ep.source,
        }


# ----------------------------------------------------------------------
# SmolVLA collator: raw views/strings -> model inputs + training targets
# ----------------------------------------------------------------------


def _squash_bilinear(frame_hwc: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear-resize one ``[H, W, 3]`` uint8 frame to ``(out_h, out_w)``."""
    if frame_hwc.shape[:2] == (out_h, out_w):
        return frame_hwc
    x = torch.from_numpy(np.ascontiguousarray(frame_hwc)).float().permute(2, 0, 1)[None]
    x = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return x.clamp(0, 255).to(torch.uint8)[0].permute(1, 2, 0).numpy()


class SmolVLAProgressCollator:
    """Turn raw transition samples into SmolVLA model inputs + RISE targets.

    Two stages, in order:

    1. **DreamZero canvas geometry**: the model trains on real frames but at
       inference scores the WAM's dreamed canvas, so raw frames are first put
       through the *frozen IDM/SFT video geometry* -- ``VideoCrop(0.95)`` then
       resize, exterior|wrist width-concat, and a squash to the 160x320 dream
       canvas, then split into two 160x160 views. This reuses
       :func:`...idm.data._build_sft_video_pipeline` verbatim so the training
       frames share the dreams' field-of-view and resolution (data README §4.1:
       *do not train at 256 and hope the resize matches -- use the pipeline
       geometry*).
    2. **Official SmolVLA preprocessing** (design §9): resize-with-pad to the
       SigLIP input size, normalize to ``[-1, 1]``, and tokenize the
       newline-terminated verbatim instruction with the checkpoint's tokenizer
       (right-padded to ``tokenizer_max_length``).

    Exterior and wrist are passed as separate views in ``[exterior, wrist]``
    order -- the same layout the model scores at inference. Build it from a
    loaded :class:`SmolVLAPolicy` so the image size, tokenizer, and padding
    match the checkpoint exactly.
    """

    def __init__(self, policy):
        from transformers import AutoTokenizer

        cfg = policy.config
        self.resize_hw = tuple(cfg.resize_imgs_with_padding or (512, 512))
        self.max_length = int(getattr(cfg, "tokenizer_max_length", 48))
        self.pad_to = getattr(cfg, "pad_language_to", "max_length")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.vlm_model_name)
        self._geom = None

    def _ensure_geom(self):
        if self._geom is None:
            from rlinf.models.embodiment.dreamzero.idm.data import (
                _build_sft_video_pipeline,
            )

            # Per-view resize to (160, 320) then a width squash to the 160x320
            # canvas -- the exact double-resample IDMChunkDataset performs.
            self._geom = _build_sft_video_pipeline(
                DREAM_VIEW_SIZE, 2 * DREAM_VIEW_SIZE, CROP_SCALE, train_aug=False
            )

    def _dream_views(
        self, ext: np.ndarray, wri: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Raw exterior/wrist ``[H, W, 3]`` -> two 160x160 dream-canvas views."""
        self._ensure_geom()
        s = DREAM_VIEW_SIZE
        out = self._geom({"video.image": ext[None], "video.wrist_image": wri[None]})
        e = np.asarray(out["video.image"])[0]  # [s, 2s, 3]
        w = np.asarray(out["video.wrist_image"])[0]  # [s, 2s, 3]
        canvas = np.concatenate([e, w], axis=1)  # [s, 4s, 3]
        canvas = _squash_bilinear(canvas, s, 2 * s)  # [s, 2s, 3] = WAM 160x320 canvas
        return canvas[:, :s], canvas[:, s:]  # exterior-left, wrist-right

    def _prep_views(
        self, ext_list: list[np.ndarray], wri_list: list[np.ndarray]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the two SmolVLA view tensors: dream geometry, then resize-pad to
        the SigLIP size and ``[-1, 1]``."""
        from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad

        pairs = [self._dream_views(e, w) for e, w in zip(ext_list, wri_list)]

        def _stack(views: list[np.ndarray]) -> torch.Tensor:
            x = torch.from_numpy(np.stack(views)).float().div_(255.0)  # [B,H,W,3]
            x = x.permute(0, 3, 1, 2).contiguous()  # [B,3,H,W]
            return resize_with_pad(x, *self.resize_hw, pad_value=0) * 2.0 - 1.0

        return _stack([p[0] for p in pairs]), _stack([p[1] for p in pairs])

    def _tokenize(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        texts = [t if t.endswith("\n") else f"{t}\n" for t in texts]
        enc = self.tokenizer(
            texts,
            padding=self.pad_to,
            padding_side="right",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"].bool()

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        bsize = len(batch)
        ext, wri = self._prep_views(
            [b["exterior"] for b in batch], [b["wrist"] for b in batch]
        )
        next_ext, next_wri = self._prep_views(
            [b["next_exterior"] for b in batch], [b["next_wrist"] for b in batch]
        )
        mask = torch.ones(bsize, dtype=torch.bool)
        lang_tokens, lang_masks = self._tokenize([b["language"] for b in batch])

        return {
            # current-state inputs (online Vθ(o_t, l))
            "images": [ext, wri],
            "img_masks": [mask, mask],
            "lang_tokens": lang_tokens,
            "lang_masks": lang_masks,
            # next-state inputs (target V(o_{t+1}, l)); language reused
            "next_images": [next_ext, next_wri],
            "next_img_masks": [mask, mask],
            # RISE training targets (windowed reward / done derived in td_loss)
            "t_index": torch.tensor([b["t_index"] for b in batch], dtype=torch.long),
            "episode_len": torch.tensor(
                [b["episode_len"] for b in batch], dtype=torch.long
            ),
            "success_mask": torch.tensor(
                [b["success"] for b in batch], dtype=torch.bool
            ),
            "source": [b["source"] for b in batch],
        }


def build_progress_episodes(
    pi_root: Optional[str] = None,
    nvidia_root: Optional[str] = None,
) -> list[ProgressEpisode]:
    """Read and concatenate the configured PI and NVIDIA episode sets."""
    episodes: list[ProgressEpisode] = []
    if pi_root:
        episodes += read_pi_episodes(pi_root)
    if nvidia_root:
        episodes += read_nvidia_episodes(nvidia_root)
    if not episodes:
        raise ValueError("No episodes: provide pi_root and/or nvidia_root.")
    return episodes
