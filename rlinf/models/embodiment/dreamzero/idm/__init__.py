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

"""Inverse Dynamics Model for the DreamZero PRM (Milestone 3).

Self-contained package: model architecture here; dataset, training, and
validation entrypoints land alongside it as the milestone progresses.
"""

from rlinf.models.embodiment.dreamzero.idm.model import (
    IDM,
    IDMConfig,
    compute_loss,
    count_parameters,
    split_canvas,
)

__all__ = ["IDM", "IDMConfig", "compute_loss", "count_parameters", "split_canvas"]

# IDMChunkDataset / collate_idm live in .data; import that module directly
# (it pulls the rlinf dataset chain, which needs the training environment).
