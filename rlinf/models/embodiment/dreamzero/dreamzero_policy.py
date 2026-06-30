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
        _, _, action_keys, language_keys = collect_dreamzero_dataset_keys(
            config.data_transforms, embodiment_tag
        )
        self._action_keys = tuple(action_keys)
        # Model-space language key in the converted obs (for the progress PRM
        # term's prompt; see _progress_language).
        self._language_key = str(language_keys[0]) if language_keys else None
        # Debug counter for save_video_pred (see _maybe_save_video_pred).
        self._video_pred_call_count = 0
        # Counter for best-of-K candidate sampling (see _predict_best_of_k).
        self._bok_call_count = 0
        # Lazily-built PRM selector and last executed env-space action
        # (cross-chunk continuity term); see _select_candidate.
        self._bok_prm = None
        self._bok_prev_action = None
        # Lazily-seeded RNG for the random-selection control baseline
        # (bok_selector=random); see _select_random.
        self._bok_random_rng = None
        self._bok_exec_alias_warned = False

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

        Which candidate is executed depends on ``bok_selector``: "random" picks
        one uniformly at random per env (control baseline, see
        ``_select_random``); "prm" delegates to the weighted PRM terms in
        ``bok_prm_terms`` (see ``rlinf.models.embodiment.dreamzero.prm`` and
        ``_select_candidate``). Candidate-0 baseline runs should use
        ``best_of_k=1`` and skip this path. All candidates are saved/logged for
        diversity verification.

        Enabled with ``actor.model.best_of_k`` > 1. Optional:
        ``bok_base_seed`` (default: the head's own seed), ``bok_output_dir``
        (default "dreamzero_best_of_k") for the diversity jsonl,
        ``bok_selector``, ``bok_prm_terms``, and the PRM knobs.
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

        need_cons = self._cons_dreams_needed()
        # A latent IDM consumes ``video_pred`` directly (no decode); a pixel IDM
        # needs the decoded RGB canvas. The kind is auto-detected by the PRM
        # from the IDM checkpoint (see DreamZeroPRM / _load_consistency_idm).
        uses_latent = bool(
            need_cons and getattr(self._ensure_prm(), "cons_uses_latent", False)
        )
        need_prog = self._prog_dreams_needed()
        candidates = []
        seeds = []
        cons_inputs = []  # per-candidate IDM inputs: raw latents or decoded RGB
        prog_dreams = []  # per-candidate decoded RGB canvases for the progress term
        try:
            for k in range(num_candidates):
                seed = base_seed + k
                action_head.seed = seed
                with torch.no_grad():
                    pred = self.lazy_joint_video_action_causal(normalized_input)
                candidates.append(pred)
                seeds.append(seed)
                video_pred = pred.get("video_pred")
                # Decode the dream once per candidate when something needs RGB:
                # the pixel-IDM consistency term and/or the progress term (which
                # always scores decoded RGB, independent of the IDM kind). The
                # decode is reused by the save hook so a chunk is never decoded
                # twice. A latent IDM consumes the raw ``video_pred`` directly.
                decoded = None
                if video_pred is not None and (
                    need_prog or (need_cons and not uses_latent)
                ):
                    decoded = self._decode_dream(video_pred)
                if need_cons and video_pred is not None:
                    cons_inputs.append(video_pred if uses_latent else decoded)
                if need_prog and decoded is not None:
                    prog_dreams.append(decoded)
                self._maybe_save_video_pred(
                    video_pred,
                    mode=mode,
                    input_images=normalized_input.get("images") if k == 0 else None,
                    candidate_index=k,
                    seed=seed,
                    increment_call=(k == num_candidates - 1),
                    decoded=decoded,
                )
        finally:
            action_head.seed = original_seed

        # Env-space actions per candidate (the real env dims; computed once,
        # used for selection and for diversity logging).
        env_actions = []
        for cand in candidates:
            cand_batch = self.unapply(
                Batch(normalized_action=cand["action_pred"].float()),
                obs=obs,
            )
            env_actions.append(np.asarray(self._actions_from_unapply(cand_batch.act)))

        chosen_index, select_info = self._select_candidate(
            env_actions,
            cons_inputs if need_cons else None,
            uses_latent,
            prog_dreams if need_prog else None,
            obs,
        )
        selected_pred = self._gather_selected_candidate(candidates, chosen_index)

        # Last executed action, for the PRM's cross-chunk continuity term.
        env_stack = np.stack(env_actions)  # [K, B, T, D]
        chosen_per_env = self._chosen_per_env(chosen_index, env_stack.shape[1])
        env_ids = np.arange(env_stack.shape[1])
        self._bok_prev_action = env_stack[chosen_per_env, env_ids, -1, :].copy()

        self._log_best_of_k_diversity(
            candidates,
            seeds,
            mode=mode,
            env_actions=env_actions,
            chosen_index=chosen_index,
            select_info=select_info,
        )
        return selected_pred

    def _cons_dreams_needed(self) -> bool:
        """Whether best-of-K must prepare the consistency input this call: only
        when the PRM selector is active and ``bok_prm_terms`` includes
        ``consistency``. The input is the raw ``video_pred`` latent for a latent
        IDM (no decode) or the decoded RGB canvas for a pixel IDM; either way it
        is always-on here, independent of ``save_video_pred`` (whose own decode
        is separately capped).
        """
        if self._bok_selector() != "prm":
            return False
        return bool(self._ensure_prm().uses_consistency)

    def _prog_dreams_needed(self) -> bool:
        """Whether best-of-K must decode dreams for the progress term this call:
        only when the PRM selector is active and ``bok_prm_terms`` includes
        ``progress``. The progress model always scores decoded RGB dreams,
        independent of the IDM's pixel/latent kind.
        """
        if self._bok_selector() != "prm":
            return False
        return bool(self._ensure_prm().uses_progress)

    def _progress_language(self, obs: dict) -> list:
        """Per-env instruction strings for the progress term.

        Robometer scores the dreamed exterior frames directly, and the current
        PRM progress term uses the last Robometer progress prediction as the
        candidate value. The only non-visual conditioning it needs is the
        verbatim LIBERO instruction, pulled from the converted obs by the
        language key.
        """
        language = obs.get(self._language_key)
        if language is None:
            ext_key = self._rollout_obs_layout.video_fields[0][1]
            language = [""] * np.asarray(obs[ext_key]).shape[0]
        return list(language)

    @staticmethod
    def _chosen_per_env(chosen_index: Any, batch_size: int) -> np.ndarray:
        """Normalize scalar/list PRM choices to one candidate id per env."""
        chosen = np.asarray(chosen_index, dtype=np.int64)
        if chosen.ndim == 0:
            return np.full((batch_size,), int(chosen), dtype=np.int64)
        chosen = chosen.reshape(-1)
        if chosen.size != batch_size:
            raise ValueError(
                f"chosen_index has {chosen.size} entries but batch has {batch_size}"
            )
        return chosen

    def _gather_selected_candidate(
        self, candidates: list[dict[str, Any]], chosen_index: Any
    ) -> dict[str, Any]:
        """Build a model output dict with per-env best-of-K selections."""
        first_action = candidates[0]["action_pred"]
        batch_size = int(first_action.shape[0])
        chosen_np = self._chosen_per_env(chosen_index, batch_size)

        # Fast path: every env chose the same candidate.
        if np.all(chosen_np == chosen_np[0]):
            return candidates[int(chosen_np[0])]

        out: dict[str, Any] = {}
        env_choice = torch.as_tensor(
            chosen_np, dtype=torch.long, device=first_action.device
        )
        for key, value in candidates[0].items():
            if torch.is_tensor(value) and value.shape[:1] == (batch_size,):
                stacked = torch.stack([cand[key] for cand in candidates], dim=0)
                gather_idx = env_choice.view(
                    1, batch_size, *([1] * (stacked.ndim - 2))
                ).expand(1, batch_size, *stacked.shape[2:])
                out[key] = stacked.gather(0, gather_idx).squeeze(0)
            else:
                # Non-batched metadata cannot be selected independently per env.
                # Keep candidate 0's value; action_pred/video_pred are batched.
                out[key] = value
        return out

    def _ensure_prm(self):
        """Lazily build the best-of-K PRM selector (loads the IDM if configured).

        Shared by ``_predict_best_of_k`` (to learn whether the consistency IDM
        is latent) and ``_select_candidate``, so the PRM -- and its IDM -- is
        constructed exactly once per policy.
        """
        if self._bok_prm is None:
            from rlinf.models.embodiment.dreamzero.prm import DreamZeroPRM

            self._bok_prm = DreamZeroPRM(self.config)
        return self._bok_prm

    def _bok_selector(self) -> str:
        """Return the explicit best-of-K selector for ``best_of_k > 1``."""
        raw_selector = getattr(self.config, "bok_selector", None)
        if raw_selector is None or str(raw_selector).strip() == "":
            raise ValueError(
                "actor.model.best_of_k > 1 requires actor.model.bok_selector "
                "to be 'random' or 'prm'. Use actor.model.best_of_k=1 for the "
                "candidate-0 baseline."
            )

        selector = str(raw_selector).lower()
        if selector == "exec":
            if not self._bok_exec_alias_warned:
                logger.warning(
                    "bok_selector='exec' is deprecated; use bok_selector='prm' "
                    "with bok_prm_terms instead."
                )
                self._bok_exec_alias_warned = True
            return "prm"
        if selector == "first":
            raise ValueError(
                "bok_selector='first' is no longer supported for best_of_k > 1. "
                "Use actor.model.best_of_k=1 for the candidate-0 baseline."
            )
        if selector not in ("random", "prm"):
            raise ValueError(
                f"Unknown bok_selector {selector!r}; use 'random' or 'prm'."
            )
        return selector

    def _select_candidate(
        self,
        env_actions: list,
        cons_inputs: Optional[list] = None,
        uses_latent: bool = False,
        prog_dreams: Optional[list] = None,
        obs: Optional[dict] = None,
    ) -> tuple[Any, Optional[dict]]:
        """Pick which best-of-K candidate to execute.

        ``bok_selector`` (Hydra ``+actor.model.bok_selector``): "random" picks
        one candidate uniformly at random per env (the control baseline that
        isolates the value of the selection metric, see ``_select_random``);
        "prm" ranks candidates with the configured PRM terms
        (:class:`rlinf.models.embodiment.dreamzero.prm.DreamZeroPRM`). The
        candidate-0 baseline is ``best_of_k=1`` and does not enter this method.

        When ``bok_prm_terms`` includes ``consistency``, ``cons_inputs`` (the
        per-candidate IDM inputs prepared in ``_predict_best_of_k``) are passed
        in ``context["dream_input"]`` -- raw WAM video latents when
        ``uses_latent`` (latent IDM), otherwise the decoded RGB canvases split
        into the IDM's per-view layout.

        When a progress checkpoint is configured (``bok_progress_model_path``),
        the per-candidate decoded dreams ``prog_dreams`` and the current ``obs``
        are passed in ``context["progress"]`` -- the dreams split into per-view
        RGB, plus the verbatim instruction language -- so the PRM adds the
        Robometer last-frame progress reward term.
        """
        selector = self._bok_selector()
        if selector == "random":
            return self._select_random(env_actions)
        if selector != "prm":
            raise ValueError(
                f"Unknown bok_selector {selector!r}; use 'random' or 'prm'."
            )
        self._ensure_prm()
        env_stack = torch.as_tensor(np.stack(env_actions), dtype=torch.float32)
        context = {}
        if self._bok_prev_action is not None:
            context["prev_action"] = self._bok_prev_action
        # Consistency term input. Latent IDM: stack the raw WAM video latents
        # [B, C, T, H, W] -> [K, B, C, T, H, W]. Pixel IDM: split each decoded
        # canvas into the per-view layout -> [K, B, V, F, 3, H, W/2].
        if self._bok_prm.cons_scorer is not None and cons_inputs:
            if uses_latent:
                context["dream_input"] = torch.stack(cons_inputs, dim=0)
            else:
                from rlinf.models.embodiment.dreamzero.idm.model import split_canvas

                context["dream_input"] = torch.stack(
                    [split_canvas(d) for d in cons_inputs], dim=0
                )
        # Progress term input: per-candidate dreams split into exterior/wrist
        # views -> [K, B, V, F, 3, H, W], plus the task language. Only when a
        # progress model is loaded.
        if (
            self._bok_prm.prog_scorer is not None
            and prog_dreams
            and obs is not None
        ):
            from rlinf.models.embodiment.dreamzero.idm.model import split_canvas

            dream_rgb = torch.stack([split_canvas(d) for d in prog_dreams], dim=0)
            language = self._progress_language(obs)
            context["progress"] = {"dream_rgb": dream_rgb, "language": language}
        chosen, info = self._bok_prm.select(env_stack, context=context)
        chosen_per_env = info.get("chosen_index_per_env")
        num_envs = len(chosen_per_env) if isinstance(chosen_per_env, list) else 1
        if num_envs > 1:
            def _shape(key):
                return list(np.asarray(info[key]).shape) if key in info else None

            logger.info(
                "[dreamzero best-of-k] call %d: selector=prm chose per-env "
                "candidates (num_envs=%d, prm_terms=%s, chosen_counts=%s, "
                "exec_pen_per_env_shape=%s, exec_score_per_env_shape=%s, "
                "cons_pen_per_env_shape=%s, cons_score_per_env_shape=%s, "
                "progress_per_env_shape=%s, combined_per_env_shape=%s)",
                self._bok_call_count,
                num_envs,
                info.get("prm_terms"),
                info.get("chosen_counts"),
                _shape("penalty_per_env"),
                _shape("exec_score_per_env"),
                _shape("cons_penalty_per_env"),
                _shape("cons_score_per_env"),
                _shape("progress_reward_per_env"),
                _shape("combined_score_per_env"),
            )
        else:
            exec_pen = info.get("exec_penalty")
            if exec_pen is None:
                exec_pen = info.get("penalty", [])
            exec_score = info.get("exec_score")
            if exec_score is None:
                exec_score = info.get("score", [])
            logger.info(
                "[dreamzero best-of-k] call %d: selector=prm chose candidate %s "
                "(prm_terms=%s, chosen_counts=%s, exec_pen=%s, exec_score=%s, "
                "cons_pen=%s, cons_score=%s, progress=%s, combined=%s)",
                self._bok_call_count,
                chosen,
                info.get("prm_terms"),
                info.get("chosen_counts"),
                [round(float(p), 6) for p in exec_pen],
                [round(float(s), 6) for s in exec_score],
                (
                    [round(float(p), 6) for p in info["cons_penalty"]]
                    if "cons_penalty" in info
                    else None
                ),
                (
                    [round(float(s), 6) for s in info["cons_score"]]
                    if "cons_score" in info
                    else None
                ),
                (
                    [round(float(s), 6) for s in info["progress_reward"]]
                    if "progress_reward" in info
                    else None
                ),
                (
                    [round(float(s), 6) for s in info["combined_score"]]
                    if "combined_score" in info
                    else None
                ),
            )
        return chosen, info

    def _select_random(self, env_actions: list) -> tuple[Any, dict]:
        """Best-of-K control baseline: pick one candidate uniformly at random.

        ``bok_selector=random``. The K candidates are generated exactly as for
        ``bok_selector=prm`` (same seeds, same diversity), but the choice ignores
        every score — no PRM, no IDM, no dream decode. This isolates the value
        of the selection *metric* from the value of merely having K samples:
        compare its success rate against PRM arms and the ``best_of_k=1``
        candidate-0 baseline. The choice is made per env.

        Reproducible via ``bok_random_seed`` (default 0); the RNG is created
        once and advances deterministically across calls, so a rerun with the
        same seed reproduces the same selections.
        """
        num_candidates = len(env_actions)
        num_envs = int(np.asarray(env_actions[0]).shape[0])
        seed = int(getattr(self.config, "bok_random_seed", 0) or 0)
        if self._bok_random_rng is None:
            self._bok_random_rng = np.random.default_rng(seed)
        chosen = self._bok_random_rng.integers(
            0, num_candidates, size=num_envs
        ).astype(np.int64)
        chosen_counts = np.bincount(chosen, minlength=num_candidates).tolist()
        info = {
            "selector": "random",
            "chosen_index_per_env": chosen.tolist(),
            "chosen_counts": chosen_counts,
            "bok_random_seed": seed,
        }
        logger.info(
            "[dreamzero best-of-k] call %d: selector=random chose %s "
            "(num_envs=%d, K=%d, chosen_counts=%s, seed=%d)",
            self._bok_call_count,
            chosen.tolist(),
            num_envs,
            num_candidates,
            chosen_counts,
            seed,
        )
        # Scalar for the single-env case; per-env list otherwise.
        chosen_index = int(chosen[0]) if num_envs == 1 else chosen.tolist()
        return chosen_index, info

    def _log_best_of_k_diversity(
        self,
        candidates: list,
        seeds: list[int],
        mode: str,
        env_actions: list,
        chosen_index: Any = 0,
        select_info: Optional[dict] = None,
    ) -> None:
        """Log and persist how different the K candidates are.

        Diversity is measured on the **env-space actions** (un-normalized via
        ``unapply``, before gripper binarization, precomputed by
        ``_predict_best_of_k``): pairwise L2 over flattened chunks of the
        real environment dims (e.g. 7 for LIBERO). The raw 32-wide
        ``action_pred`` is NOT used for stats because dims beyond the env
        width are training-masked padding with unconstrained values; it is
        only used as a bit-identical check (max pairwise L2 == 0 on the full
        output means seed variation had no effect) and a warning is logged.
        Per-candidate env-space actions, the env-space distance matrix (flat
        ``[K, K]`` only for ``B == 1``, otherwise ``[B, K, K]``), the chosen
        candidate, and (when the PRM selector ran) the per-env PRM term
        breakdown are written to
        ``<bok_output_dir>/best_of_k_info.jsonl``.
        """
        k = len(candidates)

        # Stats on the real env dims only (padding dims are meaningless).
        env_stack = torch.as_tensor(
            np.stack(env_actions), dtype=torch.float32
        )  # [K, B, T, env_dim]
        num_envs = int(env_stack.shape[1])
        iu = torch.triu_indices(k, k, offset=1)
        chosen_np = self._chosen_per_env(chosen_index, num_envs)
        chosen_counts = (
            select_info.get("chosen_counts") if select_info is not None else None
        )
        if chosen_counts is None:
            chosen_counts = np.bincount(chosen_np, minlength=k).tolist()

        # Bit-identical check on the full model output (catches broken seeds
        # even if differences were only in padded dims).
        raw_stack = torch.stack(
            [c["action_pred"].detach().float().cpu() for c in candidates]
        )

        call_index = self._bok_call_count
        if num_envs == 1:
            env_flat = env_stack[:, 0].reshape(k, -1)
            env_dist = torch.cdist(env_flat, env_flat)  # [K, K]
            env_pairwise = env_dist[iu[0], iu[1]]
            mean_l2 = env_pairwise.mean().item()
            min_l2 = env_pairwise.min().item()
            max_l2 = env_pairwise.max().item()

            raw = raw_stack[:, 0].reshape(k, -1)
            raw_identical = torch.cdist(raw, raw)[iu[0], iu[1]].max().item() == 0.0

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
            pairwise_info = {
                "pairwise_l2_env": env_dist.tolist(),
                "pairwise_l2_mean": mean_l2,
                "pairwise_l2_min": min_l2,
                "pairwise_l2_max": max_l2,
                "identical": raw_identical,
            }
        else:
            env_flat = env_stack.transpose(0, 1).contiguous().reshape(
                num_envs, k, -1
            )
            env_dist = torch.cdist(env_flat, env_flat)  # [B, K, K]
            env_pairwise = env_dist[:, iu[0], iu[1]]
            zero_pair_envs = (env_pairwise.min(dim=1).values == 0.0).tolist()

            raw = raw_stack.transpose(0, 1).contiguous().reshape(num_envs, k, -1)
            raw_dist = torch.cdist(raw, raw)
            raw_identical_envs = (
                raw_dist[:, iu[0], iu[1]].max(dim=1).values == 0.0
            ).tolist()
            raw_identical = all(raw_identical_envs)

            logger.info(
                "[dreamzero best-of-k] call %d (%s): K=%d seeds=%s num_envs=%d "
                "chosen_counts=%s env-action (%dd) pairwise L2 kept per-env "
                "with shape=%s",
                call_index,
                mode,
                k,
                seeds,
                num_envs,
                chosen_counts,
                env_stack.shape[-1],
                list(env_dist.shape),
            )
            pairwise_info = {
                "pairwise_l2_env_per_env": env_dist.tolist(),
                "pairwise_l2_env_shape": list(env_dist.shape),
                "identical": raw_identical,
                "identical_per_env": raw_identical_envs,
                "env_pairwise_has_zero_per_env": zero_pair_envs,
            }
        if raw_identical:
            logger.warning(
                "[dreamzero best-of-k] all %d candidates are bit-identical; "
                "seed variation had no effect.",
                k,
            )
        elif num_envs == 1 and min_l2 == 0.0:
            logger.warning(
                "[dreamzero best-of-k] at least one candidate pair has "
                "identical env-space actions (min pairwise L2 = 0)."
            )
        elif num_envs > 1 and any(zero_pair_envs):
            logger.warning(
                "[dreamzero best-of-k] at least one candidate pair has "
                "identical env-space actions in %d/%d envs.",
                sum(bool(x) for x in zero_pair_envs),
                num_envs,
            )

        output_dir = getattr(self.config, "bok_output_dir", "dreamzero_best_of_k")
        os.makedirs(output_dir, exist_ok=True)
        info = {
            "call_index": call_index,
            "mode": mode,
            "k": k,
            "seeds": seeds,
            "env_action_shape": list(env_stack.shape[1:]),
            "chosen_index": chosen_index,
            "chosen_index_per_env": chosen_np.tolist(),
            "chosen_counts": chosen_counts,
            "env_actions_per_candidate": [np.asarray(a).tolist() for a in env_actions],
        }
        info.update(pairwise_info)
        if select_info is not None:
            info["prm"] = select_info
        info_path = os.path.join(output_dir, "best_of_k_info.jsonl")
        with open(info_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(info) + "\n")
        self._bok_call_count += 1

    @torch.no_grad()
    def _decode_dream(self, video_pred: torch.Tensor) -> torch.Tensor:
        """Decode a WAN video latent to RGB dream frames (no gating).

        ``video_pred`` is the ``[B, C, T, H, W]`` latent from
        ``lazy_joint_video_action_causal``; returns uint8 ``[B, T, H, W, 3]``
        with the same recipe/normalization as the save hook and the IDM canvas.

        This is the **always-on** decode used by the consistency PRM term:
        unlike ``_maybe_save_video_pred`` it is never gated by
        ``save_video_pred`` or capped by ``video_pred_max_calls`` -- the
        selector needs every candidate's dream on every call. The save hook
        reuses this result so a save+consistency run never decodes a chunk
        twice.
        """
        action_head = self.action_head
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
        rgb = rearrange(frames, "B C T H W -> B T H W C")
        return ((rgb.float() + 1) * 127.5).clip(0, 255).to(torch.uint8)

    def _maybe_save_video_pred(
        self,
        video_pred: Optional[torch.Tensor],
        mode: str,
        input_images: Optional[torch.Tensor] = None,
        candidate_index: Optional[int] = None,
        seed: Optional[int] = None,
        increment_call: bool = True,
        decoded: Optional[torch.Tensor] = None,
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

        # Reuse the consistency decode if the caller already produced it;
        # otherwise decode here (this path is gated above by save_video_pred /
        # video_pred_max_calls, so it stays a capped debug decode).
        if decoded is None:
            decoded = self._decode_dream(video_pred)
        rgb = decoded.cpu().numpy()  # [B, T, H, W, 3] uint8
        # Decoded B C T H W shape, reconstructed from the RGB tensor for the
        # debug log / jsonl (matches the previous vae.decode output shape).
        decoded_shape = [
            rgb.shape[0],
            rgb.shape[4],
            rgb.shape[1],
            rgb.shape[2],
            rgb.shape[3],
        ]
        logger.info(
            "[dreamzero video debug] decoded frames B C T H W shape = %s",
            tuple(decoded_shape),
        )
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
                "decoded_shape": decoded_shape,
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
