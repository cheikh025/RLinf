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

"""Config for the RLinf wrapper around NVIDIA Cosmos Policy (RoboCasa).

This is a thin bridge: RLinf's ``actor.model`` DictConfig -> a
:class:`CosmosPolicyConfig` -> NVIDIA's own ``PolicyEvalConfig``. The wrapper
never reimplements Cosmos preprocessing; it only fills the same knobs NVIDIA's
official RoboCasa eval uses so ``get_action`` behaves identically to
``run_robocasa_eval.py`` (see
``dreamzero_docs/robocasa_cosmos_rlinf_integration_plan_README.md``).
"""

from dataclasses import dataclass, field
from typing import Any

from omegaconf import DictConfig, OmegaConf

from rlinf.utils.logging import get_logger

logger = get_logger()


@dataclass
class CosmosPolicyConfig:
    """RLinf-side config for the Cosmos Policy RoboCasa checkpoint.

    Defaults mirror NVIDIA's official RoboCasa inference config
    ``cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference`` exactly,
    with the single documented override ``flip_images=False`` (RLinf's
    ``RobocasaEnv`` already flips frames upright, so a second flip inside
    ``get_action`` would invert them).
    """

    # --- RLinf-facing ---
    model_type: str = "cosmos_policy"
    model_path: str = "nvidia/Cosmos-Policy-RoboCasa-Predict2-2B"
    precision: str = "bf16"
    action_dim: int = 7
    # Number of action steps RLinf executes per model call (open-loop horizon).
    # The model generates ``chunk_size`` (=32); we execute the first
    # ``num_action_chunks`` (=16), which reproduces NVIDIA's num_open_loop_steps.
    num_action_chunks: int = 16

    # --- Cosmos checkpoint / experiment selection ---
    cosmos_config: str = (
        "cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference"
    )
    cosmos_config_file: str = "cosmos_policy/config/config.py"
    dataset_stats_path: str = ""
    t5_text_embeddings_path: str = ""

    # --- Cosmos diffusion / IO knobs (must match training/eval) ---
    chunk_size: int = 32
    num_denoising_steps_action: int = 5
    num_denoising_steps_future_state: int = 1
    num_denoising_steps_value: int = 1
    use_wrist_image: bool = True
    num_wrist_images: int = 1
    use_proprio: bool = True
    normalize_proprio: bool = True
    unnormalize_actions: bool = True
    trained_with_image_aug: bool = True
    use_jpeg_compression: bool = True
    # OVERRIDE vs NVIDIA default (True): RLinf env already flips upright.
    flip_images: bool = False
    deterministic: bool = True
    use_variance_scale: bool = False
    num_queries_best_of_n: int = 1
    seed: int = 195

    # Extra fields captured from yaml but not part of the dataclass schema.
    extra: dict[str, Any] = field(default_factory=dict)

    # Fields forwarded to NVIDIA's PolicyEvalConfig. Keys are our attribute
    # names; values are the corresponding PolicyEvalConfig attribute names.
    # TODO(verify): confirm these names against the installed cosmos-policy
    # commit (feasibility note pinned 18a2acc). Any rename here is a one-line fix.
    _EVAL_FIELD_MAP = {
        "model_path": "ckpt_path",
        "cosmos_config": "config",
        "cosmos_config_file": "config_file",
        "dataset_stats_path": "dataset_stats_path",
        "t5_text_embeddings_path": "t5_text_embeddings_path",
        "chunk_size": "chunk_size",
        "num_action_chunks": "num_open_loop_steps",
        "num_denoising_steps_action": "num_denoising_steps_action",
        "num_denoising_steps_future_state": "num_denoising_steps_future_state",
        "num_denoising_steps_value": "num_denoising_steps_value",
        "use_wrist_image": "use_wrist_image",
        "num_wrist_images": "num_wrist_images",
        "use_proprio": "use_proprio",
        "normalize_proprio": "normalize_proprio",
        "unnormalize_actions": "unnormalize_actions",
        "trained_with_image_aug": "trained_with_image_aug",
        "use_jpeg_compression": "use_jpeg_compression",
        "flip_images": "flip_images",
        "deterministic": "deterministic",
        "use_variance_scale": "use_variance_scale",
        "num_queries_best_of_n": "num_queries_best_of_n",
        "seed": "seed",
    }

    @classmethod
    def from_dictconfig(cls, cfg: DictConfig) -> "CosmosPolicyConfig":
        """Build from RLinf ``actor.model`` (already merged with rollout paths)."""
        raw = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(raw, dict):
            raise ValueError("cosmos_policy actor.model must resolve to a mapping.")

        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for key, value in raw.items():
            if key in known:
                kwargs[key] = value
            else:
                extra[key] = value
        kwargs["extra"] = extra

        config = cls(**kwargs)
        config._validate()
        return config

    def _validate(self) -> None:
        if not self.dataset_stats_path:
            raise ValueError(
                "cosmos_policy requires actor.model.dataset_stats_path "
                "(NVIDIA robocasa_dataset_statistics.json)."
            )
        if self.use_proprio and self.normalize_proprio and not self.dataset_stats_path:
            raise ValueError(
                "normalize_proprio=True requires dataset_stats_path."
            )
        if not self.t5_text_embeddings_path:
            # Not fatal: get_action can compute T5 live, but that needs a loaded
            # T5-XXL encoder (extra VRAM). Warn loudly so eval isn't silently slow.
            logger.warning(
                "cosmos_policy: actor.model.t5_text_embeddings_path is empty; "
                "text embeddings will be computed live if the encoder is loaded."
            )
        if self.action_dim != 7:
            logger.warning(
                "cosmos_policy: action_dim=%s (RoboCasa Cosmos policy emits 7D). "
                "Verify env action_space matches.",
                self.action_dim,
            )
        if self.flip_images:
            logger.warning(
                "cosmos_policy: flip_images=True, but RLinf's RobocasaEnv already "
                "flips frames upright. This will double-flip. Set flip_images=False "
                "unless you are feeding RAW robosuite frames."
            )

    def to_policy_eval_config(self) -> Any:
        """Construct NVIDIA's ``PolicyEvalConfig`` with our knobs.

        Instantiated with defaults, then each mapped field is set via ``setattr``
        so the bridge is robust to PolicyEvalConfig gaining/losing fields across
        cosmos-policy versions. A renamed field surfaces as a clear warning.
        """
        from rlinf.models.embodiment.cosmos_policy import _import_policy_eval_config

        PolicyEvalConfig = _import_policy_eval_config()

        try:
            eval_cfg = PolicyEvalConfig()
        except TypeError:
            # Some required fields have no default; seed the core three.
            eval_cfg = PolicyEvalConfig(
                ckpt_path=self.model_path,
                config=self.cosmos_config,
                config_file=self.cosmos_config_file,
            )

        for attr, eval_attr in self._EVAL_FIELD_MAP.items():
            value = getattr(self, attr)
            if not hasattr(eval_cfg, eval_attr):
                # TODO(verify): field name drift in cosmos-policy PolicyEvalConfig.
                logger.warning(
                    "cosmos_policy: PolicyEvalConfig has no field %r "
                    "(from %r); skipping.",
                    eval_attr,
                    attr,
                )
                continue
            setattr(eval_cfg, eval_attr, value)

        # Forward any pass-through knobs the user set in yaml under actor.model
        # that happen to be real PolicyEvalConfig fields (e.g. run_id_note,
        # local_log_dir). Ignored otherwise.
        for key, value in self.extra.items():
            if hasattr(eval_cfg, key):
                setattr(eval_cfg, key, value)

        return eval_cfg
