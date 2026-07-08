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
``actor.model.model_type: cosmos_policy`` works like any other embodied model.
See ``dreamzero_docs/robocasa_cosmos_rlinf_integration_plan_README.md``.
"""

import contextlib
from typing import Any

from omegaconf import DictConfig, OmegaConf

from rlinf.utils.logging import get_logger

logger = get_logger()

_INSTALL_HINT = (
    "The `cosmos-policy` package is not importable. Install the cosmos_policy "
    "model venv via `bash requirements/install.sh embodied --model cosmos_policy "
    "--env robocasa` (clones NVlabs/cosmos-policy), and run inside that venv."
)


@contextlib.contextmanager
def _tolerate_resolver_reregistration():
    """Tolerate OmegaConf resolver name clashes during the cosmos-policy import.

    cosmos-policy re-registers OmegaConf resolvers (e.g. ``subtract``) at import
    time, and RLinf's :mod:`rlinf.utils.omega_resolver` has already registered a
    ``subtract`` resolver. ``register_new_resolver`` raises on a duplicate name
    unless ``replace=True``, so force ``replace`` for the duration of the cosmos
    import only. The semantics are compatible (cosmos's variadic ``subtract`` is a
    superset of RLinf's binary one), and RLinf's resolvers stay intact if the
    cosmos import fails partway through.
    """
    orig = OmegaConf.register_new_resolver

    def _patched(name, resolver, *args, **kwargs):
        kwargs.setdefault("replace", True)
        return orig(name, resolver, *args, **kwargs)

    OmegaConf.register_new_resolver = _patched
    try:
        yield
    finally:
        OmegaConf.register_new_resolver = orig


def _import_cosmos_utils() -> Any:
    """Lazily import NVIDIA's cosmos_utils (heavy deps; import at build time)."""
    try:
        with _tolerate_resolver_reregistration():
            from cosmos_policy.experiments.robot import cosmos_utils  # type: ignore
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(_INSTALL_HINT) from exc
    return cosmos_utils


def _import_policy_eval_config() -> Any:
    """Lazily import NVIDIA's RoboCasa ``PolicyEvalConfig``."""
    try:
        with _tolerate_resolver_reregistration():
            from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import (  # type: ignore
                PolicyEvalConfig,
            )
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(_INSTALL_HINT) from exc
    return PolicyEvalConfig


def get_model(cfg: DictConfig, torch_dtype=None):
    """Build the Cosmos Policy RoboCasa wrapper from RLinf ``actor.model`` cfg.

    ``torch_dtype`` is accepted for signature parity with other embodied models;
    Cosmos loads its own precision (bf16) inside ``get_model``.
    """
    from rlinf.models.embodiment.cosmos_policy.cosmos_config import CosmosPolicyConfig
    from rlinf.models.embodiment.cosmos_policy.cosmos_policy import CosmosPolicy

    config = CosmosPolicyConfig.from_dictconfig(cfg)
    cosmos_utils = _import_cosmos_utils()
    eval_cfg = config.to_policy_eval_config()

    logger.info(
        "cosmos_policy: loading %s (config=%s, flip_images=%s, chunk_size=%s, "
        "exec_horizon=%s)",
        config.model_path,
        config.cosmos_config,
        config.flip_images,
        config.chunk_size,
        config.num_action_chunks,
    )

    # NVIDIA's loader: returns (model, model_config); places model on device.
    cosmos_model, cosmos_model_config = cosmos_utils.get_model(eval_cfg)
    if hasattr(cosmos_model, "eval"):
        cosmos_model.eval()

    dataset_stats = cosmos_utils.load_dataset_stats(config.dataset_stats_path)

    # Initialize the (module-global) T5 embedding cache so get_action can look up
    # per-instruction embeddings. Safe to skip if the helper is absent.
    init_t5 = getattr(cosmos_utils, "init_t5_text_embeddings_cache", None)
    if callable(init_t5) and config.t5_text_embeddings_path:
        init_t5(config.t5_text_embeddings_path)

    return CosmosPolicy(
        config=config,
        cosmos_model=cosmos_model,
        cosmos_model_config=cosmos_model_config,
        dataset_stats=dataset_stats,
        eval_cfg=eval_cfg,
        cosmos_utils=cosmos_utils,
    )
