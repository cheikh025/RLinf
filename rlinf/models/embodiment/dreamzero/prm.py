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
import torch.nn.functional as F

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


def _per_env_payload(name: str, values: torch.Tensor) -> dict[str, Any]:
    """Return per-env values, plus flat candidate values only for B == 1."""
    payload = {f"{name}_per_env": values.tolist()}
    if values.shape[1] == 1:
        payload[name] = values[:, 0].tolist()
    return payload


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
    ) -> dict[str, Any]:
        """Compute executability penalties for K candidate chunks.

        Args:
            env_actions: ``[K, B, T, D]`` env-space actions (gripper last).
            prev_action: optional ``[B, D]`` last executed action of the
                previous chunk, prepended for seam continuity.

        Returns:
            Dict with ``*_per_env`` matrices shaped ``[K, B]``. For ``B == 1``,
            flat per-candidate lists are also included for concise logging.
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
        alim_pen = (violate**2).mean(dim=(2, 3))
        acc_pen = _huber(accel, self.acc_huber_delta).mean(dim=(2, 3))
        jerk_pen = _huber(jerk, self.jerk_huber_delta).mean(dim=(2, 3))

        # Gripper chattering: sign convention matches the policy binarization.
        grip_sign = torch.where(chunk[..., -1] > 0, 1.0, -1.0)  # [K, B, T]
        flips = (grip_sign[:, :, 1:] != grip_sign[:, :, :-1]).float().sum(dim=2)
        flip_pen = torch.relu(flips - 1.0)

        penalty = (
            self.w_alim * alim_pen
            + self.w_grip * flip_pen
            + self.w_acc * acc_pen
            + self.w_jerk * jerk_pen
        )
        out = {}
        out.update(_per_env_payload("alim_pen", alim_pen))
        out.update(_per_env_payload("acc_pen", acc_pen))
        out.update(_per_env_payload("jerk_pen", jerk_pen))
        out.update(_per_env_payload("flip_pen", flip_pen))
        out.update(_per_env_payload("penalty", penalty))
        return out


class ConsistencyScorer:
    """Cycle-consistency term (Milestone 3): does the WAM's action chunk agree
    with a frozen IDM's reading of the WAM's own dreamed video?

    For each best-of-K candidate, the IDM -- trained to invert a dreamed clip
    back to the action chunk that produced it -- predicts actions from the
    decoded dream; the penalty is the distance to the WAM's predicted actions
    on the 7 env dims: arm SmoothL1 in the IDM's *standardized* units (so
    translation does not drown rotation) plus gripper sign disagreement. Low
    penalty = video and actions tell the same story (internally coherent); a
    high penalty flags a candidate whose two heads disagree.

    Needs no ground truth and no simulator -- computed purely from the model's
    own outputs, exploiting DreamZero's joint video+action structure. The IDM
    is loaded once (see :meth:`IDM.from_checkpoint`) and passed in.
    """

    def __init__(
        self,
        idm,
        w_arm: float = 1.0,
        w_grip: float = 1.0,
        arm_beta: float = 0.1,
    ):
        self.idm = idm
        self.device = next(idm.parameters()).device
        self.w_arm = float(w_arm)
        self.w_grip = float(w_grip)
        self.arm_beta = float(arm_beta)

    @torch.no_grad()
    def score(
        self,
        env_actions: torch.Tensor,
        dream_videos: torch.Tensor,
    ) -> dict[str, Any]:
        """Compute consistency penalties for K candidate chunks.

        Args:
            env_actions: ``[K, B, T, D]`` env-space WAM actions (gripper last).
            dream_videos: ``[K, B, V, F, 3, H, W]`` uint8 decoded dreams in the
                IDM's split-canvas layout (V views, F frames).

        Returns:
            ``*_per_env`` matrices shaped ``[K, B]``. For ``B == 1``, flat
            per-candidate lists are also included for concise logging.
        """
        a_wam = torch.as_tensor(env_actions, dtype=torch.float32)
        if a_wam.ndim != 4:
            raise ValueError(f"env_actions must be [K, B, T, D], got {a_wam.shape}")
        k, b, t, d = a_wam.shape

        vids = torch.as_tensor(dream_videos)
        if tuple(vids.shape[:2]) != (k, b):
            raise ValueError(
                f"dream_videos must start [K={k}, B={b}, ...], got {tuple(vids.shape)}"
            )
        # One batched IDM forward over all K*B clips, then back to [K, B, T, D].
        idm_in = vids.reshape(k * b, *vids.shape[2:]).to(self.device)
        idm_act = self.idm.predict(idm_in).reshape(k, b, t, d).float().cpu()

        arm_std = self.idm.arm_std.detach().float().cpu().clamp_min(1e-6)
        diff = (a_wam[..., :-1] - idm_act[..., :-1]) / arm_std
        arm = F.smooth_l1_loss(
            diff, torch.zeros_like(diff), beta=self.arm_beta, reduction="none"
        ).mean(dim=(2, 3))  # [K, B]

        a_grip = torch.where(a_wam[..., -1] > 0, 1.0, -1.0)
        i_grip = torch.where(idm_act[..., -1] > 0, 1.0, -1.0)
        grip = (a_grip != i_grip).float().mean(dim=2)  # [K, B]

        penalty = self.w_arm * arm + self.w_grip * grip
        out = {}
        out.update(_per_env_payload("cons_arm", arm))
        out.update(_per_env_payload("cons_grip", grip))
        out.update(_per_env_payload("cons_penalty", penalty))
        return out


class DreamZeroPRM:
    """Combine PRM terms and select which best-of-K candidate to execute.

    Always scores executability (:class:`ExecutabilityScorer`). When an IDM
    checkpoint is configured (``bok_idm_model_path``) *and* the policy passes
    the decoded dreams in ``context["dream_videos"]``, the cycle-consistency
    term (:class:`ConsistencyScorer`) is added and the two are combined as
    ``exec_lambda * exec_score + cons_lambda * cons_score`` on the bounded
    (0, SCORE_MAX] axis. Without an IDM (or without dreams) it is pure
    executability and behaves exactly as before.

    Selection: pick the best candidate directly (``argmin`` penalty for
    exec-only, ``argmax`` combined score when consistency is on). Candidate 0
    has no special tie-break privilege in best-of-K mode.

    Config (read via ``getattr`` from the policy's ``DreamZeroConfig``, all
    optional Hydra ``+actor.model.*`` keys): ``bok_exec_w_alim``,
    ``bok_exec_w_grip``, ``bok_exec_w_acc``, ``bok_exec_w_jerk``; and for
    consistency ``bok_idm_model_path``, ``bok_idm_device``,
    ``bok_exec_lambda``, ``bok_cons_lambda``, ``bok_cons_arm_w``,
    ``bok_cons_grip_w``.
    """

    #: EVA's bounded score mapping (logged only; argmin of penalty is the
    #: same ranking). Bounded (0, 10] scales keep the later lambda weighting
    #: of executability vs consistency sane.
    SCORE_MAX = 10.0
    SCORE_P0 = 1.0
    SCORE_GAMMA = 0.5

    def __init__(self, config: Any = None):
        self.exec_scorer = ExecutabilityScorer(
            w_alim=float(getattr(config, "bok_exec_w_alim", 1.0)),
            w_grip=float(getattr(config, "bok_exec_w_grip", 1.0)),
            w_acc=float(getattr(config, "bok_exec_w_acc", 0.1)),
            w_jerk=float(getattr(config, "bok_exec_w_jerk", 0.05)),
        )
        # Optional cycle-consistency term (Milestone 3). Built only when an IDM
        # checkpoint is configured (``bok_idm_model_path``), so executability-
        # only runs load no IDM and behave exactly as before.
        self.cons_scorer = None
        self.exec_lambda = float(getattr(config, "bok_exec_lambda", 1.0))
        self.cons_lambda = float(getattr(config, "bok_cons_lambda", 1.0))
        idm_path = getattr(config, "bok_idm_model_path", None)
        if idm_path:
            from rlinf.models.embodiment.dreamzero.idm.model import IDM

            idm = IDM.from_checkpoint(
                str(idm_path),
                device=str(getattr(config, "bok_idm_device", "cuda")),
                dtype=torch.float32,
            )
            self.cons_scorer = ConsistencyScorer(
                idm,
                w_arm=float(getattr(config, "bok_cons_arm_w", 1.0)),
                w_grip=float(getattr(config, "bok_cons_grip_w", 1.0)),
            )

    def _bounded(self, penalty):
        """EVA bounded score: monotone-decreasing map penalty -> (0, SCORE_MAX].

        argmin of penalty == argmax of this score; using it lets the
        executability and consistency penalties (different raw scales) be
        combined on one comparable (0, SCORE_MAX] axis with the lambda weights.
        """
        return self.SCORE_MAX * (1.0 + penalty / self.SCORE_P0) ** (-self.SCORE_GAMMA)

    def select(
        self,
        env_actions: torch.Tensor,
        context: Optional[dict] = None,
    ) -> tuple[Any, dict]:
        """Pick the candidate to execute.

        Args:
            env_actions: ``[K, B, T, D]`` env-space candidate action chunks.
            context: optional extras; today only ``prev_action`` (``[B, D]``
                last executed action) is used. Milestone 3 adds the dreamed
                video here for the consistency term.

        Returns:
            ``(chosen_index, info)`` where ``chosen_index`` is an ``int`` for
            ``B == 1`` and a list of length ``B`` for parallel eval. ``info``
            holds per-env matrices and selection metadata for logging. Flat
            candidate lists are included only for ``B == 1``.
        """
        context = context or {}
        terms = self.exec_scorer.score(
            env_actions, prev_action=context.get("prev_action")
        )
        exec_pen_env = torch.as_tensor(terms["penalty_per_env"], dtype=torch.float32)
        exec_score_env = self._bounded(exec_pen_env)
        info = dict(terms)
        info["exec_score_per_env"] = exec_score_env.tolist()
        if exec_score_env.shape[1] == 1:
            info["score"] = exec_score_env[:, 0].tolist()
            info["exec_penalty"] = terms["penalty"]
            info["exec_score"] = info["score"]

        # Consistency arm: only when an IDM is loaded and the policy supplied
        # the decoded dreams. Selection then maximizes the lambda-weighted sum
        # of the bounded executability and consistency scores.
        dreams = context.get("dream_videos")
        if self.cons_scorer is not None and dreams is not None:
            cons = self.cons_scorer.score(env_actions, dreams)
            cons_pen_env = torch.as_tensor(
                cons["cons_penalty_per_env"], dtype=torch.float32
            )
            cons_score_env = self._bounded(cons_pen_env)
            combined_env = self.exec_lambda * exec_score_env + (
                self.cons_lambda * cons_score_env
            )
            chosen_tensor = torch.argmax(combined_env, dim=0)
            chosen_per_env = chosen_tensor.cpu().tolist()
            info.update(cons)
            info["cons_score_per_env"] = cons_score_env.tolist()
            info["combined_score_per_env"] = combined_env.tolist()
            if combined_env.shape[1] == 1:
                info["cons_score"] = cons_score_env[:, 0].tolist()
                info["combined_score"] = combined_env[:, 0].tolist()
            info["chosen_index_per_env"] = chosen_per_env
            info["chosen_counts"] = torch.bincount(
                chosen_tensor.cpu(), minlength=exec_pen_env.shape[0]
            ).tolist()
            chosen = chosen_per_env[0] if len(chosen_per_env) == 1 else chosen_per_env
            info["chosen_index"] = chosen
            return chosen, info

        chosen_tensor = torch.argmin(exec_pen_env, dim=0)
        chosen_per_env = chosen_tensor.cpu().tolist()
        info["chosen_index_per_env"] = chosen_per_env
        info["chosen_counts"] = torch.bincount(
            chosen_tensor.cpu(), minlength=exec_pen_env.shape[0]
        ).tolist()
        chosen = chosen_per_env[0] if len(chosen_per_env) == 1 else chosen_per_env
        info["chosen_index"] = chosen
        return chosen, info
