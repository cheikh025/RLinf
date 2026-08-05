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

"""Per-denoising-step guidance scale for the DreamZero action head.

The head's sampling loop lives in ``groot`` and reads ``self.cfg_scale`` once
per iteration::

    for index, current_timestep in enumerate(sample_scheduler.timesteps):
        ...
        flow_pred = flow_pred_uncond + self.cfg_scale * (
            flow_pred_cond - flow_pred_uncond
        )

so a *different* value can be delivered at each step without copying any of
that loop into this repo -- which matters because the installed ``groot`` is
the source of truth and a vendored copy would silently drift from it.

This module installs a data descriptor for ``cfg_scale`` on the action head's
class. The descriptor counts reads and maps read index to denoising step, so
step ``i`` sees ``schedule[i]``.

The read-to-step mapping is **not assumed**. ``cfg_scale`` may legitimately be
read more than once per step (for example a ``if self.cfg_scale != 1.0:`` guard
that skips the unconditional pass), which would scramble the schedule with no
error. Callers must therefore pass ``reads_per_step`` explicitly, measured
first with :func:`begin_candidate` in probe mode (``schedule=None``) and
checked on every candidate afterwards via :func:`end_candidate`.
"""

from typing import Optional, Sequence

__all__ = [
    "begin_candidate",
    "end_candidate",
    "install",
    "is_installed",
    "uninstall",
]


class _CfgScaleDescriptor:
    """Data descriptor returning a per-denoising-step guidance scale.

    Defining both ``__get__`` and ``__set__`` makes this a *data* descriptor,
    which takes precedence over the instance ``__dict__``. The plain scalar the
    head was constructed with is kept under :attr:`VALUE` so assignment keeps
    working normally when no schedule is active.
    """

    VALUE = "_cfg_scale_value"
    SCHEDULE = "_cfg_scale_schedule"
    READS = "_cfg_scale_reads"
    PER_STEP = "_cfg_scale_reads_per_step"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        state = obj.__dict__
        index = state.get(self.READS, 0)
        state[self.READS] = index + 1

        schedule = state.get(self.SCHEDULE)
        if not schedule:
            return state.get(self.VALUE)

        step = index // max(int(state.get(self.PER_STEP) or 1), 1)
        # Clamp rather than raise: an overflow means the read count did not
        # match the schedule length, which end_candidate reports precisely.
        # Raising here would surface as an opaque failure mid-generation.
        if step >= len(schedule):
            step = len(schedule) - 1
        return schedule[step]

    def __set__(self, obj, value) -> None:
        obj.__dict__[self.VALUE] = float(value)


def is_installed(action_head) -> bool:
    """Whether the descriptor is already installed on the head's class."""
    return isinstance(type(action_head).__dict__.get("cfg_scale"), _CfgScaleDescriptor)


def install(action_head) -> bool:
    """Install the descriptor on ``type(action_head)``.

    Args:
        action_head: The DreamZero action head (``WANPolicyHead``) instance.

    Returns:
        ``True`` if the descriptor was installed by this call, ``False`` if it
        was already present.

    Raises:
        AttributeError: If the head has no ``cfg_scale`` to preserve.
    """
    if is_installed(action_head):
        return False

    current = getattr(action_head, "cfg_scale", None)
    if current is None:
        raise AttributeError(
            "DreamZero action head has no `cfg_scale`; a per-step guidance "
            "schedule cannot be installed."
        )

    setattr(type(action_head), "cfg_scale", _CfgScaleDescriptor())
    # groot sets `self.cfg_scale = 5.0` in __init__, leaving an instance-dict
    # entry. The data descriptor already wins over it; drop it so the two
    # cannot be read as disagreeing.
    action_head.__dict__.pop("cfg_scale", None)
    action_head.__dict__[_CfgScaleDescriptor.VALUE] = float(current)
    return True


def uninstall(action_head) -> None:
    """Remove the descriptor and restore ``cfg_scale`` as a plain attribute."""
    if not is_installed(action_head):
        return
    value = action_head.__dict__.get(_CfgScaleDescriptor.VALUE)
    delattr(type(action_head), "cfg_scale")
    for key in (
        _CfgScaleDescriptor.VALUE,
        _CfgScaleDescriptor.SCHEDULE,
        _CfgScaleDescriptor.READS,
        _CfgScaleDescriptor.PER_STEP,
    ):
        action_head.__dict__.pop(key, None)
    if value is not None:
        action_head.cfg_scale = float(value)


def begin_candidate(
    action_head,
    schedule: Optional[Sequence[float]] = None,
    reads_per_step: int = 1,
) -> None:
    """Arm the head for one candidate sampling call and reset the read counter.

    Args:
        action_head: The action head instance.
        schedule: Per-denoising-step guidance scales, one entry per step. Pass
            ``None`` to probe: the static scale is served unchanged while reads
            are still counted, which is how ``reads_per_step`` is measured.
        reads_per_step: How many ``cfg_scale`` reads occur per denoising step.
            Ignored when ``schedule`` is ``None``.
    """
    state = action_head.__dict__
    state[_CfgScaleDescriptor.SCHEDULE] = (
        [float(c) for c in schedule] if schedule else None
    )
    state[_CfgScaleDescriptor.PER_STEP] = max(int(reads_per_step), 1)
    state[_CfgScaleDescriptor.READS] = 0


def end_candidate(action_head) -> int:
    """Return how many times ``cfg_scale`` was read during the last candidate."""
    return int(action_head.__dict__.get(_CfgScaleDescriptor.READS, 0))
