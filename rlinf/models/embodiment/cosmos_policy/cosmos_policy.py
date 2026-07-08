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

"""RLinf policy wrapper for NVIDIA Cosmos Policy (RoboCasa, eval-only).

Thin adapter: translate RLinf RoboCasa rollout observations into NVIDIA's
observation dict, call the official ``cosmos_utils.get_action`` (which owns all
preprocessing: JPEG, center-crop aug, proprio/action (de)normalization, T5
lookup, diffusion sampling), and translate the result back into the tuple the
RLinf rollout worker expects.

Consistency invariants (see the integration plan README):
  * images are NOT flipped here and ``flip_images=False`` in the eval config,
    because ``RobocasaEnv`` already returns upright frames.
  * proprio is reordered to Cosmos's 9D layout ``gripper(2)+pos(3)+quat(4)`` and
    passed RAW (Cosmos normalizes with dataset_stats).
  * we return the first ``num_action_chunks`` (=16) of the 32 generated steps.
"""

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.models.embodiment.cosmos_policy.cosmos_config import CosmosPolicyConfig
from rlinf.utils.logging import get_logger

logger = get_logger()


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _frame_uint8(img: Any) -> np.ndarray:
    """Return a single RGB frame as ``(H, W, 3)`` uint8.

    RLinf's RobocasaEnv yields uint8 ``[H, W, 3]`` (already vertically flipped
    to upright). We do not add a temporal axis here.

    TODO(verify): if the installed ``get_action`` requires ``(T, H, W, 3)``
    (temporal stack) rather than a single ``(H, W, 3)`` frame, wrap with
    ``arr[None]`` in ``_build_observation``. The cosmos_utils docstring documents
    ``(T, H, W, 3)``; confirm against ``run_robocasa_eval.py``'s call site.
    """
    arr = _to_numpy(img)
    if arr.dtype != np.uint8:
        # RoboCasa renders uint8; guard against float [0,1] or [0,255].
        if np.issubdtype(arr.dtype, np.floating):
            arr = (arr * 255.0 if arr.max() <= 1.0 else arr).round()
        arr = arr.clip(0, 255).astype(np.uint8)
    return arr


class CosmosPolicy(BasePolicy, nn.Module):
    """RLinf eval-only wrapper around a loaded Cosmos Policy RoboCasa model."""

    def __init__(
        self,
        config: CosmosPolicyConfig,
        cosmos_model: Any,
        cosmos_model_config: Any,
        dataset_stats: Any,
        eval_cfg: Any,
        cosmos_utils: Any,
    ):
        nn.Module.__init__(self)
        self.config = config
        # If the cosmos model is an nn.Module this registers it so .to()/.eval()
        # propagate; otherwise it is stored as a plain attribute (cosmos get_model
        # already placed it on the device).
        self.cosmos_model = cosmos_model
        self._cosmos_model_config = cosmos_model_config
        self._dataset_stats = dataset_stats
        self._eval_cfg = eval_cfg
        self._cosmos_utils = cosmos_utils

    # ------------------------------------------------------------------ #
    # Device / mode plumbing used by the rollout worker.
    # ------------------------------------------------------------------ #
    def eval(self):
        nn.Module.eval(self)
        inner_eval = getattr(self.cosmos_model, "eval", None)
        if callable(inner_eval) and not isinstance(self.cosmos_model, nn.Module):
            inner_eval()
        return self

    def to(self, *args, **kwargs):
        # nn.Module.to moves any registered submodule (incl. cosmos_model when it
        # is an nn.Module). For a non-Module cosmos model, get_model already
        # handled placement; be a no-op that returns self.
        try:
            return nn.Module.to(self, *args, **kwargs)
        except Exception:  # pragma: no cover - defensive for exotic model objects
            return self

    # ------------------------------------------------------------------ #
    # BasePolicy required hooks.
    # ------------------------------------------------------------------ #
    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "cosmos_policy is eval-only; training forward is not implemented. "
            "See the integration plan README (SFT/RL is a later milestone)."
        )

    def _build_observation(self, env_obs: dict, b: int) -> tuple[dict, str]:
        """Map RLinf RoboCasa obs[b] -> Cosmos observation dict + prompt."""
        main = env_obs.get("main_images")
        wrist = env_obs.get("wrist_images")
        extra = env_obs.get("extra_view_images")
        states = env_obs.get("states")
        prompts = env_obs.get("task_descriptions")

        if extra is None:
            raise ValueError(
                "cosmos_policy needs 3 camera views (primary/secondary/wrist) but "
                "extra_view_images is missing. Set env image_space: 3views."
            )
        if states is None:
            raise ValueError("cosmos_policy requires env 'states' (25D RoboCasa).")

        state = _to_numpy(states[b]).astype(np.float32)
        # Cosmos 9D proprio order: gripper_qpos(2) + eef_pos(3) + eef_quat(4).
        # RLinf 25D state order: eef_pos[0:3], eef_quat[3:7], gripper_qpos[7:9], ...
        # (see rlinf/envs/robocasa/utils.py ROBOCASA_STATES).
        proprio = np.concatenate(
            [state[7:9], state[0:3], state[3:7]]
        ).astype(np.float32)

        observation = {
            "primary_image": _frame_uint8(main[b]),      # agentview_left
            "secondary_image": _frame_uint8(extra[b]),   # agentview_right
            "wrist_image": _frame_uint8(wrist[b]),        # eye_in_hand
            "proprio": proprio,
        }
        prompt = "" if prompts is None else str(prompts[b])
        return observation, prompt

    def _call_get_action(self, observation: dict, prompt: str) -> dict:
        """Invoke NVIDIA's ``get_action``.

        TODO(verify): confirm the exact positional/keyword signature against the
        installed cosmos-policy commit. Documented shape is
        ``get_action(cfg, model, dataset_stats, observation, task_description)``.
        If the T5 cache or model config must be passed explicitly (rather than
        read from a module global set by ``init_t5_text_embeddings_cache``), add
        them here.
        """
        return self._cosmos_utils.get_action(
            self._eval_cfg,
            self.cosmos_model,
            self._dataset_stats,
            observation,
            prompt,
        )

    @torch.no_grad()
    def predict_action_batch(
        self, env_obs: dict, mode: str = "eval", **kwargs
    ) -> tuple[np.ndarray, dict]:
        """Run Cosmos inference for a batch of RoboCasa envs.

        Returns ``(actions[B, num_action_chunks, 7], result)`` where ``result``
        carries Cosmos's native value estimate in ``prev_values``.
        """
        main = env_obs.get("main_images")
        if main is None:
            raise ValueError("cosmos_policy requires env 'main_images'.")
        batch_size = int(_to_numpy(main).shape[0])

        horizon = int(self.config.num_action_chunks)
        action_dim = int(self.config.action_dim)

        actions_all = []
        values_all = []
        for b in range(batch_size):
            observation, prompt = self._build_observation(env_obs, b)
            out = self._call_get_action(observation, prompt)

            act = _to_numpy(out["actions"]).astype(np.float32)  # [chunk_size, 7]
            # Execute only the first `horizon` steps (open-loop); RLinf re-queries
            # after the env steps this chunk.
            act = act[:horizon]
            if act.shape[0] != horizon:
                raise ValueError(
                    f"cosmos_policy expected >= {horizon} action steps, got "
                    f"{act.shape[0]}. Check chunk_size/num_open_loop_steps."
                )
            if act.shape[-1] != action_dim:
                raise ValueError(
                    f"cosmos_policy expected action_dim {action_dim}, got "
                    f"{act.shape[-1]}."
                )
            actions_all.append(act)

            value = out.get("value_prediction")
            values_all.append(0.0 if value is None else float(_to_numpy(value)))

        actions = np.stack(actions_all, axis=0)  # [B, horizon, 7]

        flat = torch.as_tensor(actions, dtype=torch.float32).reshape(batch_size, -1)
        result = {
            "prev_logprobs": torch.zeros(
                (batch_size, horizon * action_dim), dtype=torch.float32
            ),
            "prev_values": torch.as_tensor(
                values_all, dtype=torch.float32
            ).reshape(batch_size, 1),
            "forward_inputs": {"action": flat},
        }
        return actions, result
