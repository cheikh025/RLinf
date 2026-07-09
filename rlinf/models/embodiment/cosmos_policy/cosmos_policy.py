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

"""RLinf eval-only policy wrapper for NVIDIA Cosmos Policy (RoboCasa).

Thin adapter: translate RLinf RoboCasa rollout observations into Cosmos's
observation dict, call ``cosmos_utils.get_action`` (which owns all preprocessing:
JPEG, center-crop aug, proprio/action (de)normalization, T5 lookup, diffusion
sampling, latent extraction), and translate the result into the tuple the RLinf
rollout worker expects.

Verified contract (see the integration runbook):
  * Images are NOT flipped here (``RobocasaEnv`` already returns upright frames).
  * Proprio is reordered to Cosmos's 9D layout ``gripper(2)+pos(3)+quat(4)`` and
    passed RAW; Cosmos normalizes it with the dataset stats.
  * Cameras map main->primary, extra_view->secondary, wrist->wrist.
  * We return the first ``num_action_chunks`` of the generated ``chunk_size`` steps.
"""

from typing import Any

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
    """Return a single RGB frame as ``(H, W, 3)`` uint8 for Cosmos."""
    arr = _to_numpy(img)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = arr * 255.0 if arr.max() <= 1.0 else arr
        arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    return arr


class CosmosPolicy(BasePolicy, nn.Module):
    """RLinf eval-only wrapper around a loaded Cosmos Policy RoboCasa model."""

    def __init__(
        self,
        config: CosmosPolicyConfig,
        cosmos_model: nn.Module,
        cosmos_model_config: Any,
        dataset_stats: Any,
        cosmos_cfg: Any,
        cosmos_utils: Any,
    ):
        nn.Module.__init__(self)
        self.config = config
        # Registering the Cosmos model as a submodule makes .to()/.eval()/.parameters()
        # propagate through the standard nn.Module machinery the rollout worker uses.
        self.cosmos_model = cosmos_model
        self._cosmos_model_config = cosmos_model_config
        self._dataset_stats = dataset_stats
        self._cosmos_cfg = cosmos_cfg
        self._cosmos_utils = cosmos_utils

    # ------------------------------------------------------------------ #
    # BasePolicy required hooks.
    # ------------------------------------------------------------------ #
    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "cosmos_policy is eval-only; a training/RL forward is not implemented."
        )

    def _build_observation(self, env_obs: dict, b: int) -> tuple[dict, str]:
        """Map RLinf RoboCasa ``env_obs[b]`` -> Cosmos observation dict + prompt."""
        main = env_obs.get("main_images")
        wrist = env_obs.get("wrist_images")
        extra = env_obs.get("extra_view_images")
        states = env_obs.get("states")
        prompts = env_obs.get("task_descriptions")

        if main is None or wrist is None:
            raise ValueError(
                "cosmos_policy requires env 'main_images' and 'wrist_images'."
            )
        if extra is None:
            raise ValueError(
                "cosmos_policy needs 3 camera views (primary/secondary/wrist) but "
                "extra_view_images is missing. Set env image_space: 3views."
            )
        if states is None:
            raise ValueError("cosmos_policy requires env 'states' (25D RoboCasa).")

        state = _to_numpy(states[b]).astype(np.float32)
        # Cosmos 9D proprio: gripper_qpos(2) + eef_pos(3) + eef_quat(4).
        # RLinf 25D state: eef_pos[0:3], eef_quat[3:7], gripper_qpos[7:9]
        # (see rlinf/envs/robocasa/utils.py ROBOCASA_STATES).
        proprio = np.concatenate(
            [state[7:9], state[0:3], state[3:7]]
        ).astype(np.float32)

        observation = {
            "primary_image": _frame_uint8(main[b]),  # agentview_left
            "secondary_image": _frame_uint8(extra[b]),  # agentview_right
            "wrist_image": _frame_uint8(wrist[b]),  # eye_in_hand
            "proprio": proprio,
        }
        prompt = "" if prompts is None else str(prompts[b])
        return observation, prompt

    @torch.no_grad()
    def predict_action_batch(
        self, env_obs: dict, mode: str = "eval", **kwargs
    ) -> tuple[np.ndarray, dict]:
        """Run Cosmos inference for a batch of RoboCasa envs.

        Returns ``(actions[B, num_action_chunks, 7], result)``. Cosmos generates
        actions per single observation, so we loop over the batch (its ``batch_size``
        arg tiles one observation for best-of-N, not distinct envs).
        """
        main = env_obs.get("main_images")
        if main is None:
            raise ValueError("cosmos_policy requires env 'main_images'.")
        batch_size = int(_to_numpy(main).shape[0])

        horizon = int(self.config.num_action_chunks)
        action_dim = int(self.config.action_dim)
        compute_value = bool(self.config.compute_value)

        actions_all = []
        values_all = []
        for b in range(batch_size):
            observation, prompt = self._build_observation(env_obs, b)
            out = self._cosmos_utils.get_action(
                self._cosmos_cfg,
                self.cosmos_model,
                self._dataset_stats,
                observation,
                prompt,
                seed=int(self.config.seed),
                randomize_seed=False,
                num_denoising_steps_action=int(
                    self.config.num_denoising_steps_action
                ),
                generate_future_state_and_value_in_parallel=compute_value,
            )

            act = np.asarray(out["actions"], dtype=np.float32)  # [chunk_size, 7]
            act = act[:horizon]
            if act.shape[0] != horizon:
                raise ValueError(
                    f"cosmos_policy expected >= {horizon} action steps, got "
                    f"{act.shape[0]}. Check chunk_size / num_action_chunks."
                )
            if act.shape[-1] != action_dim:
                raise ValueError(
                    f"cosmos_policy expected action_dim {action_dim}, got "
                    f"{act.shape[-1]}."
                )
            actions_all.append(act)

            value = out.get("value_prediction") if compute_value else None
            values_all.append(0.0 if value is None else float(np.asarray(value)))

        actions = np.stack(actions_all, axis=0)  # [B, horizon, 7]

        flat = (
            torch.as_tensor(actions, dtype=torch.float32)
            .reshape(batch_size, -1)
            .cpu()
        )
        result = {
            "prev_logprobs": torch.zeros_like(flat, dtype=torch.float32),
            "prev_values": torch.as_tensor(
                values_all, dtype=torch.float32
            ).reshape(batch_size, 1),
            "forward_inputs": {"action": flat},
        }
        return actions, result
