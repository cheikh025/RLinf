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

"""Process Reward Model (PRM) for DreamZero best-of-K candidate selection.

Phase 1 (Milestone 2): executability-only scoring, EVA-style
(github.com/RobbinW/EVA, ``flow_grpo/idm_reward.py``) adapted to delta
actions. The cycle-consistency term (frozen IDM on the dreamed video) is a
later milestone and slots into :class:`DreamZeroPRM` without touching the
policy. Design and verified constants:
``dreamzero_prm_milestone2_executability_README.md``.
"""

from typing import Any, Optional

import torch

# Hard per-step acceleration bounds in normalized action units, derived from
# Franka Panda rate limits (libfranka rate_limiting.h: 13 m/s^2 translational,
# 25 rad/s^2 rotational) at robosuite defaults (20 Hz, OSC_POSE scaling
# 0.05 m / 0.5 rad per unit): 13/400/0.05 = 0.65, 25/400/0.5 = 0.125.
# Verified against a successful LIBERO episode (max observed |diff|:
# translation 0.542, rotation 0.045) -- the bounds never fire on good
# behavior. Velocity violations are impossible (actions clipped to [-1, 1])
# and jerk violations are unreachable (bound 16.25 vs max expressible 4.0),
# so acceleration is the only spec-bindable hard term.
DEFAULT_ACC_BOUNDS = (0.65, 0.65, 0.65, 0.125, 0.125, 0.125)


def _huber(x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """EVA's Huber loss: quadratic below ``delta``, linear above."""
    ax = x.abs()
    return torch.where(ax <= delta, 0.5 * (x**2), delta * (ax - 0.5 * delta))


class ExecutabilityScorer:
    """Score env-space action chunks for physical executability (r_exec).

    Operates on delta actions ``[K, B, T, D]`` where the first ``D - 1`` dims
    are arm commands (LIBERO OSC_POSE: 3 translation + 3 rotation, normalized
    per-step deltas) and the last dim is the gripper.

    Delta actions are velocity-level commands, so the kinematic chain is
    shifted one derivative vs EVA (whose IDM outputs joint positions)::

        v = actions          (commanded velocity, up to scale)
        a = diff(actions)    (acceleration, per-step units)
        j = diff2(actions)   (jerk)

    Terms (per candidate, arm dims only unless noted):

    - ``alim_pen``: ``relu(|a| - acc_bounds)^2`` mean -- the only hard,
      spec-bindable violation term (see ``DEFAULT_ACC_BOUNDS``).
    - ``acc_pen`` / ``jerk_pen``: Huber smoothness on ``a`` / ``j`` with
      deltas tied to the bounds (EVA pattern) -- soft terms, kept light to
      avoid the smoothest-of-K conservatism bias.
    - ``flip_pen``: gripper chattering, ``max(0, sign_flips - 1)`` over the
      chunk. The sim gripper takes only ``sign(action)`` (robosuite
      ``PandaGripper``), so kinematics on this dim are meaningless; one flip
      is a deliberate grasp/release, more is unexecutable fluttering. Signs
      follow the policy's binarization rule (``> 0``).

    If ``prev_action`` (the last executed action of the previous chunk) is
    given, it is prepended so the cross-chunk seam is scored too.
    """

    def __init__(
        self,
        w_alim: float = 1.0,
        w_grip: float = 1.0,
        w_acc: float = 0.1,
        w_jerk: float = 0.05,
        acc_bounds: tuple = DEFAULT_ACC_BOUNDS,
        acc_huber_scale: float = 0.5,
        jerk_huber_scale: float = 1.0,
    ):
        self.w_alim = float(w_alim)
        self.w_grip = float(w_grip)
        self.w_acc = float(w_acc)
        self.w_jerk = float(w_jerk)
        self.acc_bounds = torch.as_tensor(acc_bounds, dtype=torch.float32)
        self.acc_huber_delta = self.acc_bounds * float(acc_huber_scale)
        self.jerk_huber_delta = self.acc_bounds * float(jerk_huber_scale)

    def score(
        self,
        env_actions: torch.Tensor,
        prev_action: Optional[torch.Tensor] = None,
    ) -> dict[str, list[float]]:
        """Compute executability penalties for K candidate chunks.

        Args:
            env_actions: ``[K, B, T, D]`` env-space actions (gripper last).
            prev_action: optional ``[B, D]`` last executed action of the
                previous chunk, prepended for seam continuity.

        Returns:
            Dict of per-candidate lists (length K): ``alim_pen``, ``acc_pen``,
            ``jerk_pen``, ``flip_pen``, ``penalty`` (weighted total).
        """
        chunk = torch.as_tensor(env_actions, dtype=torch.float32)
        if chunk.ndim != 4:
            raise ValueError(f"env_actions must be [K, B, T, D], got {chunk.shape}")
        k = chunk.shape[0]
        num_arm = chunk.shape[-1] - 1
        if num_arm != self.acc_bounds.shape[0]:
            raise ValueError(
                f"acc_bounds has {self.acc_bounds.shape[0]} dims but actions "
                f"have {num_arm} arm dims; configure bounds for this embodiment."
            )

        if prev_action is not None:
            prev = torch.as_tensor(prev_action, dtype=torch.float32)
            prev = prev.reshape(1, chunk.shape[1], 1, chunk.shape[3]).expand(
                k, -1, -1, -1
            )
            chunk = torch.cat([prev, chunk], dim=2)

        arm = chunk[..., :num_arm]
        accel = arm[:, :, 1:] - arm[:, :, :-1]  # [K, B, T-1, num_arm]
        jerk = accel[:, :, 1:] - accel[:, :, :-1]

        violate = torch.relu(accel.abs() - self.acc_bounds)
        alim_pen = (violate**2).mean(dim=(1, 2, 3))
        acc_pen = _huber(accel, self.acc_huber_delta).mean(dim=(1, 2, 3))
        jerk_pen = _huber(jerk, self.jerk_huber_delta).mean(dim=(1, 2, 3))

        # Gripper chattering: sign convention matches the policy binarization.
        grip_sign = torch.where(chunk[..., -1] > 0, 1.0, -1.0)  # [K, B, T]
        flips = (grip_sign[:, :, 1:] != grip_sign[:, :, :-1]).float().sum(dim=2)
        flip_pen = torch.relu(flips - 1.0).mean(dim=1)

        penalty = (
            self.w_alim * alim_pen
            + self.w_grip * flip_pen
            + self.w_acc * acc_pen
            + self.w_jerk * jerk_pen
        )
        return {
            "alim_pen": alim_pen.tolist(),
            "acc_pen": acc_pen.tolist(),
            "jerk_pen": jerk_pen.tolist(),
            "flip_pen": flip_pen.tolist(),
            "penalty": penalty.tolist(),
        }


class DreamZeroPRM:
    """Combine PRM terms and select which best-of-K candidate to execute.

    Today this wraps only :class:`ExecutabilityScorer`; the cycle-consistency
    term (Milestone 3, frozen IDM on the dreamed video) will be added here
    and fed through ``context`` -- the policy-side hook does not change.

    Selection: ``argmin`` of the total penalty with a tie-break toward
    candidate 0 (the baseline seed): candidate ``k != 0`` is chosen only if
    it beats candidate 0 by more than ``select_margin``, so selection
    deviates from baseline behavior only on a real preference.

    Config (read via ``getattr`` from the policy's ``DreamZeroConfig``, all
    optional Hydra ``+actor.model.*`` keys): ``bok_select_margin``,
    ``bok_exec_w_alim``, ``bok_exec_w_grip``, ``bok_exec_w_acc``,
    ``bok_exec_w_jerk``.
    """

    #: EVA's bounded score mapping (logged only; argmin of penalty is the
    #: same ranking). Bounded (0, 10] scales keep the later lambda weighting
    #: of executability vs consistency sane.
    SCORE_MAX = 10.0
    SCORE_P0 = 1.0
    SCORE_GAMMA = 0.5

    def __init__(self, config: Any = None):
        self.select_margin = float(getattr(config, "bok_select_margin", 0.0) or 0.0)
        self.exec_scorer = ExecutabilityScorer(
            w_alim=float(getattr(config, "bok_exec_w_alim", 1.0)),
            w_grip=float(getattr(config, "bok_exec_w_grip", 1.0)),
            w_acc=float(getattr(config, "bok_exec_w_acc", 0.1)),
            w_jerk=float(getattr(config, "bok_exec_w_jerk", 0.05)),
        )

    def select(
        self,
        env_actions: torch.Tensor,
        context: Optional[dict] = None,
    ) -> tuple[int, dict]:
        """Pick the candidate to execute.

        Args:
            env_actions: ``[K, B, T, D]`` env-space candidate action chunks.
            context: optional extras; today only ``prev_action`` (``[B, D]``
                last executed action) is used. Milestone 3 adds the dreamed
                video here for the consistency term.

        Returns:
            ``(chosen_index, info)`` where ``info`` holds the per-candidate
            term breakdown, scores, and selection metadata for logging.
        """
        context = context or {}
        terms = self.exec_scorer.score(
            env_actions, prev_action=context.get("prev_action")
        )
        penalty = terms["penalty"]
        chosen = int(min(range(len(penalty)), key=penalty.__getitem__))
        margin_vs_cand0 = penalty[0] - penalty[chosen]
        if chosen != 0 and margin_vs_cand0 <= self.select_margin:
            chosen = 0
        scores = [
            self.SCORE_MAX * (1.0 + p / self.SCORE_P0) ** (-self.SCORE_GAMMA)
            for p in penalty
        ]
        info = dict(terms)
        info["score"] = scores
        info["chosen_index"] = chosen
        info["margin_vs_cand0"] = margin_vs_cand0
        return chosen, info
