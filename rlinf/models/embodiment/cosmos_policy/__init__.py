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

"""RLinf integration of NVIDIA Cosmos Policy (RoboCasa), eval-only.

Wraps the official ``cosmos-policy`` inference stack behind RLinf's
:class:`~rlinf.models.embodiment.base_policy.BasePolicy` so that
``actor.model.model_type: cosmos_policy`` runs like any other embodied model.
"""

from typing import Any

from omegaconf import DictConfig

from rlinf.utils.logging import get_logger

logger = get_logger()

_INSTALL_HINT = (
    "The `cosmos_policy` package is not importable. Install the cosmos_policy "
    "model into the RLinf embodied venv via `bash requirements/install.sh embodied "
    "--model cosmos_policy --env robocasa` and run inside that venv."
)


def _import_cosmos_utils() -> Any:
    """Lazily import NVIDIA's cosmos_utils (heavy deps; only at model build)."""
    try:
        from cosmos_policy.experiments.robot import cosmos_utils  # type: ignore
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(_INSTALL_HINT) from exc
    return cosmos_utils


def get_model(cfg: DictConfig, torch_dtype=None):
    """Build the Cosmos Policy RoboCasa wrapper from RLinf ``actor.model`` cfg.

    ``torch_dtype`` is accepted for signature parity with other embodied models;
    Cosmos loads its own precision (bf16) inside ``get_model``.
    """
    from rlinf.models.embodiment.cosmos_policy.cosmos_config import CosmosPolicyConfig
    from rlinf.models.embodiment.cosmos_policy.cosmos_policy import CosmosPolicy

    config = CosmosPolicyConfig.from_dictconfig(cfg)
    cosmos_utils = _import_cosmos_utils()
    cosmos_cfg = config.to_cosmos_cfg()

    logger.info(
        "cosmos_policy: loading %s (experiment=%s, chunk_size=%s, exec_horizon=%s, "
        "compute_value=%s)",
        config.model_path,
        config.cosmos_config,
        config.chunk_size,
        config.num_action_chunks,
        config.compute_value,
    )

    # NVIDIA's loader returns (model, model_config) and places the model on device.
    cosmos_model, cosmos_model_config = cosmos_utils.get_model(cosmos_cfg)
    cosmos_model.eval()

    dataset_stats = cosmos_utils.load_dataset_stats(config.dataset_stats_path)

    # Preload the (module-global) T5 embedding cache so get_action can look up
    # per-instruction embeddings without loading the T5 encoder at rollout time.
    if config.t5_text_embeddings_path:
        cosmos_utils.init_t5_text_embeddings_cache(config.t5_text_embeddings_path)

    return CosmosPolicy(
        config=config,
        cosmos_model=cosmos_model,
        cosmos_model_config=cosmos_model_config,
        dataset_stats=dataset_stats,
        cosmos_cfg=cosmos_cfg,
        cosmos_utils=cosmos_utils,
    )
