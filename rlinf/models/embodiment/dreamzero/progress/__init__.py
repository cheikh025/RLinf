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

"""SmolVLA-based progress value model for the DreamZero PRM.

Self-contained package: the model architecture (Milestone 1) lives here; the
dataset (Milestone 2, ``data.py``) and the RISE training entrypoint
(Milestone 3, ``train.py``) land alongside it as the milestones progress.
Design record: ``dreamzero_docs/dreamzero_progress_model_design_README.md``.
"""

from rlinf.models.embodiment.dreamzero.progress.losses import (
    DEFAULT_GAMMA,
    DEFAULT_TAU,
    EMATarget,
    progress_regression_loss,
    progress_target,
    td_loss,
    td_target,
)
from rlinf.models.embodiment.dreamzero.progress.model import (
    ProgressModelConfig,
    SmolVLAProgressModel,
    build_value_head,
    count_parameters,
    make_value_query,
)

__all__ = [
    "ProgressModelConfig",
    "SmolVLAProgressModel",
    "build_value_head",
    "count_parameters",
    "make_value_query",
    "EMATarget",
    "progress_regression_loss",
    "progress_target",
    "td_loss",
    "td_target",
    "DEFAULT_GAMMA",
    "DEFAULT_TAU",
]

# The PI/NVIDIA readers, ProgressTransitionDataset, and SmolVLAProgressCollator
# live in .data; import that module directly (it pulls pyarrow/h5py/lerobot,
# which need the training environment).
