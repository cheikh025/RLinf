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

import json
import logging
import os
from typing import Any, Optional

import numpy as np
import torch
from einops import rearrange
from groot.vla.model.dreamzero.base_vla import VLA
from tianshou.data import Batch

from rlinf.data.datasets.dreamzero.data_transforms import (
    collect_dreamzero_dataset_keys,
    convert_rollout_env_obs,
    rollout_obs_layout_for_embodiment,
)
from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.dreamzero.dreamzero_config import DreamZeroConfig
from rlinf.utils.logging import get_logger

logger = get_logger()


class DreamZeroPolicy(VLA, BasePolicy):
    """Lightweight DreamZero action model: IdentityBackbone + WANPolicyHead."""

    # CausalWanModel has to be wrapped to avoid a FSDP2 bug
    # when using with gradient checkpointing
    _no_split_modules = [
        "T5SelfAttention",  # text encoder
        "AttentionBlock",  # vae
        "CausalWanModel",  # action head
        "CausalWanAttentionBlock",  # action head layer
    ]

    def __init__(
        self,
        config: DreamZeroConfig,
    ):
        super().__init__(config)
        self.config = config
        embodiment_tag = config.embodiment_tag
        if embodiment_tag is None:
            raise ValueError(
                "DreamZeroPolicy requires config.embodiment_tag (set in get_model)."
            )
        self._rollout_obs_layout = rollout_obs_layout_for_embodiment(embodiment_tag)
        _, _, action_keys, _ = collect_dreamzero_dataset_keys(
            config.data_transforms, embodiment_tag
        )
        self._action_keys = tuple(action_keys)
        # Debug counter for save_video_pred (see _maybe_save_video_pred).
        self._video_pred_call_count = 0
        # Counter for best-of-K candidate sampling (see _predict_best_of_k).
        self._bok_call_count = 0

    # This method is called in FSDPModelManager.setup_model_and_optimizer
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={}):
        try:
            diffusion_model = getattr(getattr(self, "action_head", None), "model", None)
            enabled = True
            use_reentrant = gradient_checkpointing_kwargs.get("use_reentrant", True)

            if diffusion_model is None:
                raise ValueError("DreamZero policy must have action_head.")

            if hasattr(diffusion_model, "_set_gradient_checkpointing"):
                diffusion_model._set_gradient_checkpointing(diffusion_model, enabled)
            elif hasattr(diffusion_model, "gradient_checkpointing"):
                diffusion_model.gradient_checkpointing = enabled

            setattr(
                diffusion_model, "gradient_checkpointing_use_reentrant", use_reentrant
            )

            logging.warning(
                "DreamZero gradient checkpointing is enabled. If you encounter errors "
                "or memory leaks, consider: (1) upgrading to PyTorch 2.10 or later; "
                "(2) using use_reentrant=True to avoid issues when CUDA graphs and "
                "gradient checkpointing are used together."
            )

        except Exception:
            pass

    def apply(self, batch: Batch, **kwargs) -> Batch:
        """Run the forward modality pipeline on rollout observations.

        Input ``batch.obs`` is already in DreamZero modality keys (e.g.
        ``video.image``, ``state.state``, language key) from
        ``_observation_convert``. This method delegates to
        ``config.data_transforms``, built in ``get_model`` from Hydra cfg and
        ``metadata.json`` (via ``load_dreamzero_dataset_metadata`` +
        ``data_transforms.set_metadata``).

        Pipeline (libero_sim example, see ``libero_sim._build_composed_transform``):

        1. Video / state / action preprocessing and normalization
           (``StateActionTransform`` uses q99 stats from metadata).
        2. ``ConcatTransform.apply``: concat per-key tensors into flat
           ``state`` / ``action`` vectors. Per-key widths come from metadata
           (e.g. ``action.actions`` shape ``[7]`` for Libero).
        3. ``DreamTransform.apply``: pad state/action to ``max_state_dim`` /
           ``max_action_dim`` (typically 32 from yaml) so the WAN action head
           always sees a fixed width. Extra padded dims are zeros and masked
           during training; at inference the model still outputs width 32.

        The returned ``batch.normalized_obs`` is the dict consumed by
        ``lazy_joint_video_action_causal`` (tokens, video, padded actions, etc.).
        """
        obs = batch.obs
        normalized_input = self.config.data_transforms(obs)
        batch.normalized_obs = normalized_input
        return batch

    def unapply(self, batch: Batch, obs: Optional[dict] = None, **kwargs):
        """Invert model actions back to environment-scale per-modality tensors.

        ``batch.normalized_action`` is ``action_pred`` from the WAN head, shape
        ``[..., max_action_dim]`` (e.g. 32), matching the padded width from
        ``DreamTransform.apply``. Environment DOF is smaller (e.g. Libero 7);
        that width is **not** taken from Hydra ``action_dim`` on the policy—it
        comes from ``metadata.json`` loaded at build time:

        - ``get_model`` calls ``data_transforms.set_metadata(metadata)``.
        - ``ConcatTransform.set_metadata`` sets ``action_dims["action.actions"]``
          from ``metadata.modalities.action.<key>.shape[0]`` (7 for libero_sim).
        - On ``unapply``, transforms run in reverse order:
          ``DreamTransform.unapply`` (passthrough) →
          ``ConcatTransform.unapply`` slices ``[..., 0:env_dim]`` per
          ``action_concat_order`` → ``StateActionTransform.unapply`` reverses
          q99 normalization.

        Output is a dict like ``{"action.actions": tensor}`` with **env** width
        (7 for Libero). ``predict_action_batch`` then merges keys via
        ``_actions_from_unapply`` for the sim.

        If ``relative_action`` / ``relative_action_per_horizon`` is enabled,
        optionally adds the last ``state.*`` from ``obs`` (converted rollout
        obs passed from ``predict_action_batch``) to obtain absolute actions.
        """
        unnormalized_action = self.config.data_transforms.unapply(
            {"action": batch.normalized_action.cpu()}
        )

        # Check if relative_action is enabled and convert relative to absolute
        relative_action = self.config.relative_action
        relative_action_per_horizon = self.config.relative_action_per_horizon
        relative_action_keys = self.config.relative_action_keys
        if (
            (relative_action or relative_action_per_horizon)
            and relative_action_keys
            and obs is not None
        ):
            for key in relative_action_keys:
                action_key = f"action.{key}"
                state_key = f"state.{key}"

                if action_key not in unnormalized_action:
                    continue

                # Try to find the state data - check multiple possible key formats
                last_state = None

                # Format 1: Direct key like "state.joint_position"
                if state_key in obs:
                    last_state = obs[state_key]
                else:
                    # Format 2: Search for keys containing both "state" and the key name
                    for obs_key in obs.keys():
                        if "state" in obs_key and key in obs_key:
                            last_state = obs[obs_key]
                            break

                    # Format 3: If key is "joint_position" and obs has "state" key directly
                    # This handles cases where the observation uses modality-level keys
                    if last_state is None and "state" in obs:
                        state_data = obs["state"]
                        # Check if the state data shape matches the action shape
                        action_dim = unnormalized_action[action_key].shape[-1]
                        if torch.is_tensor(state_data):
                            state_dim = state_data.shape[-1]
                        elif isinstance(state_data, np.ndarray):
                            state_dim = state_data.shape[-1]
                        else:
                            state_dim = None

                        if state_dim == action_dim:
                            last_state = state_data

                if last_state is None:
                    continue

                if torch.is_tensor(last_state):
                    last_state = last_state.cpu().numpy()

                # Shape is (B, T, D) or (T, D), we want the last timestep
                # After indexing: (B, D) or (D,)
                if len(last_state.shape) >= 2:
                    last_state = last_state[..., -1, :]  # Get the last timestep

                # Action shape is (horizon, D) or (B, horizon, D)
                # Expand dims to broadcast: (D,) -> (1, D) or (B, D) -> (B, 1, D)
                if len(unnormalized_action[action_key].shape) > len(last_state.shape):
                    last_state = np.expand_dims(
                        last_state, axis=-2
                    )  # Add horizon dimension

                # Add state to relative action to get absolute action
                unnormalized_action[action_key] = (
                    unnormalized_action[action_key] + last_state
                )

        batch.act = unnormalized_action
        return batch

    def _process_batch(self, batch: Batch) -> dict[str, Any]:
        """Process batch."""
        # Normalize / transform
        batch = self.apply(batch)
        normalized_input = batch.normalized_obs
        # If the normalized input is still a Batch, flatten it into a pure dict
        if isinstance(normalized_input, Batch):
            normalized_input = normalized_input.__getstate__()
        # Do dtype cast if needed
        target_dtype = next(self.parameters()).dtype
        for k, v in normalized_input.items():
            if (
                torch.is_tensor(v)
                and v.dtype == torch.float32
                and target_dtype != torch.float32
            ):
                normalized_input[k] = v.to(dtype=target_dtype)
        return normalized_input

    def _observation_convert(self, env_obs: dict) -> dict:
        """Map RLinf rollout observations to DreamZero modality keys."""
        return convert_rollout_env_obs(self.config.embodiment_tag, env_obs)

    def _actions_from_unapply(self, act_dict: dict[str, Any]) -> np.ndarray:
        """Concatenate per-key unnormalized actions in dataset concat order."""
        parts: list[np.ndarray] = []
        for key in self._action_keys:
            if key not in act_dict:
                raise KeyError(
                    f"Unnormalized action missing {key!r}; "
                    f"available keys: {sorted(act_dict)}."
                )
            value = act_dict[key]
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            parts.append(np.asarray(value))
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts, axis=-1)

    def predict_action_batch(self, env_obs, mode, **kwargs) -> np.ndarray:
        """
        input:
            env_obs:
                - main_images: [B,H,W,C] uint8
                - wrist_images: [B,H,W,C] (optional, embodiment-specific)
                - extra_view_images: [B,N,H,W,C] (optional, e.g. oxe_droid)
                - states: [B,D]
                - task_descriptions: list[str] or None
        output:
            actions: np.ndarray [B, num_action_chunks, action_dim]
            result: dict  # compatible with rollout interface"""

        converted_obs = self._observation_convert(env_obs)
        batch = Batch(obs=converted_obs)
        # ---------- DreamZero inference ----------
        normalized_input = self._process_batch(batch)
        best_of_k = int(getattr(self.config, "best_of_k", 1) or 1)
        if best_of_k > 1:
            model_pred = self._predict_best_of_k(
                normalized_input,
                mode=mode,
                num_candidates=best_of_k,
                obs=converted_obs,
            )
        else:
            with torch.no_grad():
                model_pred = self.lazy_joint_video_action_causal(normalized_input)

            self._maybe_save_video_pred(
                model_pred.get("video_pred"),
                mode=mode,
                input_images=normalized_input.get("images"),
            )

        normalized_action = model_pred["action_pred"].float()

        batch = self.unapply(
            Batch(normalized_action=normalized_action),
            obs=converted_obs,
        )
        actions = self._actions_from_unapply(batch.act)

        if self._rollout_obs_layout.binarize_gripper:
            actions[..., -1] = np.where(actions[..., -1] > 0, 1.0, -1.0).astype(
                actions.dtype
            )

        flat = (
            torch.as_tensor(actions, dtype=torch.float32)
            .reshape(actions.shape[0], -1)
            .cpu()
        )
        forward_inputs = {"action": flat}
        result = {
            "prev_logprobs": torch.zeros_like(flat, dtype=torch.float32),
            "prev_values": torch.zeros((flat.shape[0], 1), dtype=torch.float32),
            "forward_inputs": forward_inputs,
        }
        return actions, result

    def _predict_best_of_k(
        self,
        normalized_input: dict[str, Any],
        mode: str,
        num_candidates: int,
        obs: dict,
    ):
        """Sample K independent (action, video) candidates by varying the seed.

        Milestone 1 of best-of-K (see dreamzero_prm_best_of_k_README.md): the
        action head seeds its diffusion noise with a fixed ``seed`` attribute,
        so K calls with the same seed are bit-identical. Candidate ``k`` uses
        ``base_seed + k``; candidate 0 keeps the head's original seed (unless
        ``bok_base_seed`` overrides it), so the executed behavior is exactly
        the single-sample path. RLinf feeds a single frame per call, which
        resets the head's stream state (``current_start_frame``/KV caches) at
        the start of every call, so the K samplings are independent.

        The executed result is always candidate 0; the other candidates are
        only saved/logged for diversity verification (no scorer yet).

        Enabled with ``actor.model.best_of_k`` > 1. Optional:
        ``bok_base_seed`` (default: the head's own seed), ``bok_output_dir``
        (default "dreamzero_best_of_k") for the diversity jsonl.
        """
        action_head = self.action_head
        if not hasattr(action_head, "seed"):
            raise AttributeError(
                "DreamZero action head has no `seed` attribute; "
                "best_of_k requires it to vary the diffusion noise."
            )
        original_seed = int(action_head.seed)
        base_seed = getattr(self.config, "bok_base_seed", None)
        base_seed = original_seed if base_seed is None else int(base_seed)

        candidates = []
        seeds = []
        try:
            for k in range(num_candidates):
                seed = base_seed + k
                action_head.seed = seed
                with torch.no_grad():
                    pred = self.lazy_joint_video_action_causal(normalized_input)
                candidates.append(pred)
                seeds.append(seed)
                self._maybe_save_video_pred(
                    pred.get("video_pred"),
                    mode=mode,
                    input_images=normalized_input.get("images") if k == 0 else None,
                    candidate_index=k,
                    seed=seed,
                    increment_call=(k == num_candidates - 1),
                )
        finally:
            action_head.seed = original_seed

        self._log_best_of_k_diversity(candidates, seeds, mode=mode, obs=obs)
        return candidates[0]

    def _log_best_of_k_diversity(
        self,
        candidates: list,
        seeds: list[int],
        mode: str,
        obs: dict,
    ) -> None:
        """Log and persist how different the K candidates are.

        Diversity is measured on the **env-space actions** (un-normalized via
        ``unapply``, before gripper binarization): pairwise L2 over flattened
        chunks of the real environment dims (e.g. 7 for LIBERO). The raw
        32-wide ``action_pred`` is NOT used for stats because dims beyond the
        env width are training-masked padding with unconstrained values; it
        is only used as a bit-identical check (max pairwise L2 == 0 on the
        full output means seed variation had no effect) and a warning is
        logged. Per-candidate env-space actions and the env-space distance
        matrix are written to ``<bok_output_dir>/best_of_k_info.jsonl``.
        """
        k = len(candidates)

        env_actions = []
        for cand in candidates:
            cand_batch = self.unapply(
                Batch(normalized_action=cand["action_pred"].float()),
                obs=obs,
            )
            env_actions.append(np.asarray(self._actions_from_unapply(cand_batch.act)))

        # Stats on the real env dims only (padding dims are meaningless).
        env_stack = torch.as_tensor(
            np.stack(env_actions), dtype=torch.float32
        )  # [K, B, T, env_dim]
        env_flat = env_stack.reshape(k, -1)
        env_dist = torch.cdist(env_flat, env_flat)  # [K, K]
        iu = torch.triu_indices(k, k, offset=1)
        env_pairwise = env_dist[iu[0], iu[1]]
        mean_l2 = env_pairwise.mean().item()
        min_l2 = env_pairwise.min().item()
        max_l2 = env_pairwise.max().item()

        # Bit-identical check on the full model output (catches broken seeds
        # even if differences were only in padded dims).
        raw = torch.stack(
            [c["action_pred"].detach().float().cpu() for c in candidates]
        ).reshape(k, -1)
        raw_identical = torch.cdist(raw, raw)[iu[0], iu[1]].max().item() == 0.0

        call_index = self._bok_call_count
        logger.info(
            "[dreamzero best-of-k] call %d (%s): K=%d seeds=%s "
            "env-action (%dd) pairwise L2 mean=%.4f min=%.4f max=%.4f",
            call_index,
            mode,
            k,
            seeds,
            env_stack.shape[-1],
            mean_l2,
            min_l2,
            max_l2,
        )
        if raw_identical:
            logger.warning(
                "[dreamzero best-of-k] all %d candidates are bit-identical; "
                "seed variation had no effect.",
                k,
            )
        elif min_l2 == 0.0:
            logger.warning(
                "[dreamzero best-of-k] at least one candidate pair has "
                "identical env-space actions (min pairwise L2 = 0)."
            )

        output_dir = getattr(self.config, "bok_output_dir", "dreamzero_best_of_k")
        os.makedirs(output_dir, exist_ok=True)
        info = {
            "call_index": call_index,
            "mode": mode,
            "k": k,
            "seeds": seeds,
            "env_action_shape": list(env_stack.shape[1:]),
            "pairwise_l2_env": env_dist.tolist(),
            "pairwise_l2_mean": mean_l2,
            "pairwise_l2_min": min_l2,
            "pairwise_l2_max": max_l2,
            "identical": raw_identical,
            "env_actions_per_candidate": [a.tolist() for a in env_actions],
        }
        info_path = os.path.join(output_dir, "best_of_k_info.jsonl")
        with open(info_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(info) + "\n")
        self._bok_call_count += 1

    def _maybe_save_video_pred(
        self,
        video_pred: Optional[torch.Tensor],
        mode: str,
        input_images: Optional[torch.Tensor] = None,
        candidate_index: Optional[int] = None,
        seed: Optional[int] = None,
        increment_call: bool = True,
    ) -> None:
        """Decode and save DreamZero's dreamed video chunk as MP4 (debug only).

        Enabled with ``actor.model.save_video_pred=true``. ``video_pred`` is the
        WAN VAE latent ``[B, C, T, H, W]`` returned by
        ``lazy_joint_video_action_causal``; it is decoded with
        ``action_head.vae.decode`` (same recipe as
        ``dreamzero/eval_utils/serve_dreamzero_wan22.py``) and written as
        ``<mode>_call_<N>.mp4`` under ``actor.model.video_pred_output_dir``.
        Tensor shapes (latent, decoded, RGB width/height/frames) are logged and
        appended to ``video_pred_info.jsonl`` for later verification.
        Optional: ``video_pred_max_calls`` caps saved calls,
        ``video_pred_fps`` sets MP4 fps (default 5, as in DreamZero serving).

        With best-of-K (``_predict_best_of_k``), ``candidate_index``/``seed``
        tag each candidate (``_cand<k>`` filename suffix, extra jsonl fields)
        and ``increment_call`` advances the call counter only once per
        env step (after the last candidate), so ``video_pred_max_calls``
        still counts env steps, not files.
        """
        if not getattr(self.config, "save_video_pred", False):
            return
        if video_pred is None:
            logger.warning(
                "save_video_pred is enabled but the model returned no video_pred."
            )
            return
        max_calls = getattr(self.config, "video_pred_max_calls", None)
        if max_calls is not None and self._video_pred_call_count >= int(max_calls):
            return

        try:
            import imageio
        except ImportError as exc:
            raise ImportError(
                "imageio is required to save dreamed videos; install rlinf[embodied]."
            ) from exc

        output_dir = getattr(
            self.config, "video_pred_output_dir", "dreamzero_video_pred"
        )
        fps = int(getattr(self.config, "video_pred_fps", 5))
        os.makedirs(output_dir, exist_ok=True)

        if input_images is not None:
            logger.info(
                "[dreamzero video debug] normalized_input['images'].shape = %s",
                tuple(input_images.shape),
            )
        logger.info(
            "[dreamzero video debug] video_pred latent shape = %s",
            tuple(video_pred.shape),
        )

        action_head = self.action_head
        with torch.no_grad():
            frames = action_head.vae.decode(
                video_pred,
                tiled=action_head.tiled,
                tile_size=(
                    action_head.tile_size_height,
                    action_head.tile_size_width,
                ),
                tile_stride=(
                    action_head.tile_stride_height,
                    action_head.tile_stride_width,
                ),
            )
        logger.info(
            "[dreamzero video debug] decoded frames B C T H W shape = %s",
            tuple(frames.shape),
        )

        rgb = rearrange(frames, "B C T H W -> B T H W C")
        rgb = ((rgb.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        logger.info(
            "[dreamzero video debug] decoded RGB B T H W C shape = %s", rgb.shape
        )

        call_index = self._video_pred_call_count
        cand_suffix = "" if candidate_index is None else f"_cand{candidate_index}"
        info_path = os.path.join(output_dir, "video_pred_info.jsonl")
        for env_idx in range(rgb.shape[0]):
            suffix = "" if rgb.shape[0] == 1 else f"_env{env_idx}"
            mp4_path = os.path.join(
                output_dir, f"{mode}_call_{call_index:06d}{cand_suffix}{suffix}.mp4"
            )
            imageio.mimsave(mp4_path, list(rgb[env_idx]), fps=fps, codec="libx264")
            info = {
                "call_index": call_index,
                "mode": mode,
                "env_index": env_idx,
                "mp4_path": mp4_path,
                "latent_shape": list(video_pred.shape),
                "decoded_shape": list(frames.shape),
                "num_frames": int(rgb.shape[1]),
                "height": int(rgb.shape[2]),
                "width": int(rgb.shape[3]),
                "channels": int(rgb.shape[4]),
                "fps": fps,
            }
            if candidate_index is not None:
                info["candidate_index"] = int(candidate_index)
            if seed is not None:
                info["seed"] = int(seed)
            with open(info_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(info) + "\n")
            logger.info("[dreamzero video debug] saved mp4 path = %s", mp4_path)
        if increment_call:
            self._video_pred_call_count += 1

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        elif forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        else:
            raise NotImplementedError

    def sft_forward(self, data=None, **kwargs):
        # Mark the start of each training iteration so PyTorch knows when
        # to reclaim memory held by CUDA graphs from the previous iteration.
        torch.compiler.cudagraph_mark_step_begin()

        if data is None:
            data = kwargs.get("data")
        if data is None:
            raise ValueError("sft_forward requires `data` from the SFT dataloader.")
        outputs = super().forward(data)
        if hasattr(outputs, "data"):
            outputs = outputs.data
        if "loss" not in outputs:
            raise ValueError("sft_forward requires `loss` in the outputs.")
        return dict(outputs)

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        """Default forward pass."""
        raise NotImplementedError
