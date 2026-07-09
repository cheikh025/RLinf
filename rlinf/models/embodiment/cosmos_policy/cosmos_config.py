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

"""Config bridge: RLinf ``actor.model`` DictConfig -> NVIDIA Cosmos Policy cfg.

The wrapper never reimplements Cosmos preprocessing. It builds a minimal config
object exposing *exactly* the attributes that ``cosmos_utils.get_model`` and
``cosmos_utils.get_action`` read, so those functions behave identically to
NVIDIA's official ``run_robocasa_eval.py``.

Design note: we deliberately do NOT import ``run_robocasa_eval.PolicyEvalConfig``.
Importing that module pulls ``import robosuite`` (+ the eval harness) at model
import time, which is exactly the surface where Cosmos's robosuite fork can clash
with RLinf's RoboCasa stack. A plain namespace avoids that entirely.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from omegaconf import DictConfig, OmegaConf

from rlinf.utils.logging import get_logger

logger = get_logger()


@dataclass
class CosmosPolicyConfig:
    """RLinf-side config for the Cosmos Policy RoboCasa checkpoint (eval-only).

    Defaults mirror NVIDIA's RoboCasa inference config
    ``cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference``. The single
    intentional deviation from a raw robosuite eval is that images are NOT flipped
    here: RLinf's ``RobocasaEnv`` already returns upright frames.
    """

    # --- RLinf execution contract ---
    model_type: str = "cosmos_policy"
    model_path: str = "nvidia/Cosmos-Policy-RoboCasa-Predict2-2B"
    precision: str = "bf16"
    action_dim: int = 7
    # Steps executed open-loop per model call (first N of the generated chunk).
    num_action_chunks: int = 16

    # --- Cosmos experiment / checkpoint selection ---
    # Experiment name registered in cosmos_policy/config/config.py.
    cosmos_config: str = "cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference"
    # Module whose make_config() registers the robocasa experiment (NOT the
    # _src/.../video2world/config.py default, which does not register it).
    cosmos_config_file: str = "cosmos_policy/config/config.py"

    # --- Normalization / text-embedding assets (ship in the HF checkpoint repo) ---
    dataset_stats_path: str = ""
    t5_text_embeddings_path: str = ""

    # --- Cosmos diffusion / IO knobs (must match training/eval) ---
    suite: str = "robocasa"
    chunk_size: int = 32
    num_denoising_steps_action: int = 5
    use_wrist_image: bool = True
    num_wrist_images: int = 1
    use_third_person_image: bool = True
    num_third_person_images: int = 2
    use_proprio: bool = True
    normalize_proprio: bool = True
    unnormalize_actions: bool = True
    trained_with_image_aug: bool = True
    use_jpeg_compression: bool = True
    use_variance_scale: bool = False
    seed: int = 195
    # If True, also run Cosmos's parallel future-image + value generation (an
    # extra VAE decode per call). Off by default: eval only needs the action, and
    # skipping it roughly halves per-step cost. When True, the value estimate is
    # surfaced in the rollout result's ``prev_values``.
    compute_value: bool = False
    # Documented no-op: RLinf env pre-flips; Cosmos get_action does not read this.
    flip_images: bool = False

    # Fields present in the yaml but not part of this schema.
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dictconfig(cls, cfg: DictConfig) -> "CosmosPolicyConfig":
        """Build from the merged RLinf ``actor.model`` DictConfig."""
        raw = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(raw, dict):
            raise ValueError("cosmos_policy actor.model must resolve to a mapping.")

        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for key, value in raw.items():
            (kwargs if key in known else extra)[key] = value
        kwargs["extra"] = extra

        config = cls(**kwargs)
        config._validate()
        return config

    def _validate(self) -> None:
        if not self.dataset_stats_path:
            raise ValueError(
                "cosmos_policy requires actor.model.dataset_stats_path "
                "(NVIDIA robocasa_dataset_statistics.json, local path or HF repo path)."
            )
        if not self.t5_text_embeddings_path:
            logger.warning(
                "cosmos_policy: actor.model.t5_text_embeddings_path is empty. "
                "get_action will compute T5 embeddings live, which requires the "
                "T5-XXL encoder to be loadable (extra VRAM). Provide the "
                "checkpoint's robocasa_t5_embeddings.pkl to avoid this."
            )
        if self.action_dim != 7:
            logger.warning(
                "cosmos_policy: action_dim=%s but RoboCasa Cosmos emits 7D "
                "([6 pose + gripper]). Verify env action_space: 7d.",
                self.action_dim,
            )
        if self.num_action_chunks > self.chunk_size:
            raise ValueError(
                f"num_action_chunks ({self.num_action_chunks}) cannot exceed "
                f"chunk_size ({self.chunk_size})."
            )

    def to_cosmos_cfg(self) -> SimpleNamespace:
        """Return the namespace consumed by ``get_model`` / ``get_action``.

        Only the attributes those functions actually read are populated (see
        cosmos_policy/experiments/robot/cosmos_utils.py).
        """
        return SimpleNamespace(
            # get_model
            ckpt_path=self.model_path,
            config=self.cosmos_config,
            config_file=self.cosmos_config_file,
            # get_action / prepare_images_for_model
            suite=self.suite,
            chunk_size=self.chunk_size,
            use_wrist_image=self.use_wrist_image,
            num_wrist_images=self.num_wrist_images,
            use_third_person_image=self.use_third_person_image,
            num_third_person_images=self.num_third_person_images,
            use_proprio=self.use_proprio,
            normalize_proprio=self.normalize_proprio,
            unnormalize_actions=self.unnormalize_actions,
            trained_with_image_aug=self.trained_with_image_aug,
            use_jpeg_compression=self.use_jpeg_compression,
            use_variance_scale=self.use_variance_scale,
        )
