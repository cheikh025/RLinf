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

import numpy as np
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
    dream (decoded RGB for a pixel IDM, or the raw WAM video latent for a
    latent IDM); the penalty is the distance to the WAM's predicted actions
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
        dream_input: torch.Tensor,
    ) -> dict[str, Any]:
        """Compute consistency penalties for K candidate chunks.

        Args:
            env_actions: ``[K, B, T, D]`` env-space WAM actions (gripper last).
            dream_input: ``[K, B, ...]`` per-candidate IDM input -- the decoded
                dream in split-canvas layout ``[K, B, V, F, 3, H, W]`` for a
                pixel IDM, or the raw WAM video latent ``[K, B, C, T, H, W]``
                for a latent IDM. The trailing dims are fed to the IDM as-is.

        Returns:
            ``*_per_env`` matrices shaped ``[K, B]``. For ``B == 1``, flat
            per-candidate lists are also included for concise logging.
        """
        a_wam = torch.as_tensor(env_actions, dtype=torch.float32)
        if a_wam.ndim != 4:
            raise ValueError(f"env_actions must be [K, B, T, D], got {a_wam.shape}")
        k, b, t, d = a_wam.shape

        clips = torch.as_tensor(dream_input)
        if tuple(clips.shape[:2]) != (k, b):
            raise ValueError(
                f"dream_input must start [K={k}, B={b}, ...], got {tuple(clips.shape)}"
            )
        # One batched IDM forward over all K*B clips, then back to [K, B, T, D].
        idm_in = clips.reshape(k * b, *clips.shape[2:]).to(self.device)
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


def _load_consistency_idm(path: str, device: str):
    """Load the consistency IDM, auto-detecting pixel vs latent.

    The checkpoint's ``idm_cfg`` distinguishes them: a latent config
    (:class:`...idm.latent_model.LatentIDMConfig`) carries ``latent_channels``,
    a pixel config (:class:`...idm.model.IDMConfig`) does not. Returns
    ``(model, is_latent)``; the model is frozen, eval, fp32. Resolves a
    checkpoint directory the same way as the ``from_checkpoint`` helpers
    (``best.pt`` > ``final.pt`` > ``latest.pt``).
    """
    import os

    ckpt_path = path
    if os.path.isdir(path):
        for name in ("best.pt", "final.pt", "latest.pt"):
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                ckpt_path = candidate
                break
        else:
            raise FileNotFoundError(
                f"No best.pt/final.pt/latest.pt in IDM checkpoint dir {path!r}"
            )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "idm_cfg" not in ckpt or "model" not in ckpt:
        raise KeyError(
            f"IDM checkpoint {ckpt_path!r} missing 'idm_cfg'/'model'."
        )
    cfg = ckpt["idm_cfg"]
    is_latent = "latent_channels" in cfg
    if is_latent:
        from rlinf.models.embodiment.dreamzero.idm.latent_model import (
            LatentActionIDM,
            LatentIDMConfig,
        )

        model = LatentActionIDM(LatentIDMConfig(**cfg))
    else:
        from rlinf.models.embodiment.dreamzero.idm.model import IDM, IDMConfig

        model = IDM(IDMConfig(**cfg))
    model.load_state_dict(ckpt["model"])
    model = model.eval().to(device=device, dtype=torch.float32).requires_grad_(False)
    return model, is_latent


def _load_progress_model(path: str, device: str):
    """Load the frozen SmolVLA progress value model for PRM scoring (§10).

    Returns the model eval/frozen on ``device`` (the ``from_checkpoint``
    contract). A directory is resolved via the checkpoint's ``progress.pt``.
    """
    from rlinf.models.embodiment.dreamzero.progress.model import (
        SmolVLAProgressModel,
    )

    return SmolVLAProgressModel.from_checkpoint(str(path), device=str(device))


class ProgressScorer:
    """Signed progress reward ``r_progress`` for best-of-K candidates (§10).

    Scores each candidate's dreamed future with the trained SmolVLA progress
    value model ``Vθ(o, l) ∈ [-1, 1]`` and returns the advance toward task
    completion relative to the conditioning frame::

        r_progress(c_k, l) = mean_{j=1..F} Vθ(ô_{k,t+j}, l) − Vθ(o_t, l)

    The conditioning value ``Vθ(o_t, l)`` is computed once and shared across the
    K candidates. Pixels are preprocessed exactly as in training via the shared
    :class:`...progress.data.SmolVLAProgressCollator`: the real conditioning
    frame goes through the full dream-geometry + SmolVLA pipeline, while the
    dreamed frames are already in the WAM canvas geometry so they take only the
    SmolVLA image stage (``[0, 255]`` -> resize-with-pad -> ``[-1, 1]``).
    """

    def __init__(
        self,
        model,
        num_future_frames: int = 8,
        max_frames_per_call: int = 0,
    ):
        from rlinf.models.embodiment.dreamzero.progress.data import (
            SmolVLAProgressCollator,
        )

        self.model = model
        self.device = next(model.parameters()).device
        # Reuse the training collator so tokenization + conditioning-frame
        # preprocessing match the checkpoint exactly.
        self.collator = SmolVLAProgressCollator(model.policy)
        self.resize_hw = self.collator.resize_hw
        self.num_future = int(num_future_frames)
        self.max_frames_per_call = int(max_frames_per_call)

    def _smolvla_prep(self, frames_b3hw: torch.Tensor) -> torch.Tensor:
        """SmolVLA image stage on already-canvas dream frames (pixels [0, 255]);
        mirrors the collator's stage-2 (resize-with-pad to SigLIP, ``[-1, 1]``)."""
        from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad

        x = frames_b3hw.to(torch.float32).div(255.0)
        return resize_with_pad(x, *self.resize_hw, pad_value=0) * 2.0 - 1.0

    def _value(self, images, lang_tokens, lang_masks) -> torch.Tensor:
        masks = [
            torch.ones(images[0].shape[0], dtype=torch.bool, device=self.device)
            for _ in images
        ]
        use_cuda = str(self.device).startswith("cuda")
        with torch.autocast(
            device_type="cuda" if use_cuda else "cpu", dtype=torch.bfloat16
        ):
            v = self.model(images, masks, lang_tokens, lang_masks)
        return v.float()

    @torch.no_grad()
    def score(
        self,
        dream_rgb: torch.Tensor,
        cond_ext: Any,
        cond_wri: Any,
        language: list,
    ) -> dict[str, Any]:
        """Per-candidate signed progress reward.

        Args:
            dream_rgb: ``[K, B, 2, F, 3, H, W]`` per-candidate decoded dream split
                into exterior/wrist views, pixels in ``[0, 255]``. Frame 0 is the
                conditioning obs ``ô_t`` (chunk offset 0); only the future frames
                ``1..F-1`` are averaged (the design's ``j = 1..8``).
            cond_ext / cond_wri: ``[B, H0, W0, 3]`` raw current-obs exterior /
                wrist frames (the conditioning state ``o_t``).
            language: list of ``B`` verbatim LIBERO instruction strings.

        Returns:
            ``progress_reward_per_env`` ``[K, B]`` (signed) plus
            ``progress_future_per_env`` and ``progress_cond_per_env`` for logging;
            flat per-candidate lists for ``B == 1``.
        """
        dream = torch.as_tensor(dream_rgb, dtype=torch.float32)
        if dream.ndim != 7 or dream.shape[2] != 2:
            raise ValueError(
                f"dream_rgb must be [K, B, 2, F, 3, H, W], got {tuple(dream.shape)}"
            )
        k, b, _v, f, c, h, w = dream.shape
        # Frame 0 of the WAM dream is the conditioning observation o_t (chunk
        # offset 0; idm/data.py num_frames=9 at offsets 0, 3, ..., 24). The
        # progress reward averages the *future* frames j=1..8 only (design §10),
        # so drop frame 0 -- including it would fold ô_t (≈ o_t) into the mean
        # and partly cancel the − Vθ(o_t) term.
        n_future = max(f - 1, 0)
        nf = min(self.num_future or n_future, n_future)
        dream = dream[:, :, :, 1:1 + nf]  # future frames j = 1..nf

        # Language + conditioning value: computed once, shared across all K.
        lt, lm = self.collator._tokenize(list(language))
        lt = lt.to(self.device)
        lm = lm.to(self.device)
        c_ext, c_wri = self.collator._prep_views(
            [np.asarray(x) for x in cond_ext], [np.asarray(x) for x in cond_wri]
        )
        v_cond = self._value(
            [c_ext.to(self.device), c_wri.to(self.device)], lt, lm
        )  # [B]

        # Future frames: [K,B,2,nf,3,H,W] -> per-view [K*B*nf,3,H,W] SmolVLA prep.
        ext = self._smolvla_prep(
            dream[:, :, 0].reshape(k * b * nf, c, h, w).to(self.device)
        )
        wri = self._smolvla_prep(
            dream[:, :, 1].reshape(k * b * nf, c, h, w).to(self.device)
        )
        ltf = lt[None, :, None].expand(k, b, nf, -1).reshape(k * b * nf, -1)
        lmf = lm[None, :, None].expand(k, b, nf, -1).reshape(k * b * nf, -1)
        n = ext.shape[0]
        step = self.max_frames_per_call or n
        outs = [
            self._value([ext[i:i + step], wri[i:i + step]], ltf[i:i + step], lmf[i:i + step])
            for i in range(0, n, step)
        ]
        v_future = torch.cat(outs, dim=0).reshape(k, b, nf).mean(dim=2)  # [K, B]

        reward = v_future - v_cond[None, :]  # [K, B]
        out = {"progress_cond_per_env": v_cond.tolist()}
        out.update(_per_env_payload("progress_future", v_future))
        out.update(_per_env_payload("progress_reward", reward))
        return out


class DreamZeroPRM:
    """Combine PRM terms and select which best-of-K candidate to execute.

    Always scores executability (:class:`ExecutabilityScorer`). When an IDM
    checkpoint is configured (``bok_idm_model_path``) *and* the policy passes
    the per-candidate dream in ``context["dream_input"]`` (decoded RGB for a
    pixel IDM, or the raw video latent for a latent IDM -- auto-detected from
    the checkpoint), the cycle-consistency term
    (:class:`ConsistencyScorer`) is added. When a progress checkpoint is
    configured (``bok_progress_model_path``) *and* the policy supplies the
    dreams + conditioning + language in ``context["progress"]``, the signed
    progress reward (:class:`ProgressScorer`, design §10) is added too. The
    active terms combine as ``exec_lambda * exec_score + cons_lambda *
    cons_score + prog_lambda * r_progress``. Without any extra term it is pure
    executability and behaves exactly as before.

    Selection: pick the best candidate directly (``argmin`` penalty for
    exec-only, ``argmax`` combined score when any extra term is on). Candidate 0
    has no special tie-break privilege in best-of-K mode.

    Config (read via ``getattr`` from the policy's ``DreamZeroConfig``, all
    optional Hydra ``+actor.model.*`` keys): ``bok_exec_w_alim``,
    ``bok_exec_w_grip``, ``bok_exec_w_acc``, ``bok_exec_w_jerk``; for
    consistency ``bok_idm_model_path``, ``bok_idm_device``,
    ``bok_exec_lambda``, ``bok_cons_lambda``, ``bok_cons_arm_w``,
    ``bok_cons_grip_w``; and for progress ``bok_progress_model_path``,
    ``bok_progress_device``, ``bok_prog_lambda``, ``bok_progress_num_frames``,
    ``bok_progress_chunk``.
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
        self.cons_uses_latent = False
        self.exec_lambda = float(getattr(config, "bok_exec_lambda", 1.0))
        self.cons_lambda = float(getattr(config, "bok_cons_lambda", 1.0))
        idm_path = getattr(config, "bok_idm_model_path", None)
        if idm_path:
            idm, self.cons_uses_latent = _load_consistency_idm(
                str(idm_path),
                str(getattr(config, "bok_idm_device", "cuda")),
            )
            self.cons_scorer = ConsistencyScorer(
                idm,
                w_arm=float(getattr(config, "bok_cons_arm_w", 1.0)),
                w_grip=float(getattr(config, "bok_cons_grip_w", 1.0)),
            )

        # Optional progress reward term (Milestone 4). Built only when a progress
        # checkpoint is configured (``bok_progress_model_path``), so runs without
        # it behave exactly as before. Mixed into the combined score with
        # ``bok_prog_lambda`` -- the same weighted-sum scheme as the other terms.
        self.prog_scorer = None
        self.prog_lambda = float(getattr(config, "bok_prog_lambda", 1.0))
        prog_path = getattr(config, "bok_progress_model_path", None)
        if prog_path:
            self.prog_scorer = ProgressScorer(
                _load_progress_model(
                    str(prog_path),
                    str(getattr(config, "bok_progress_device", "cuda")),
                ),
                num_future_frames=int(getattr(config, "bok_progress_num_frames", 8)),
                max_frames_per_call=int(getattr(config, "bok_progress_chunk", 0)),
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

        # Combined score: executability, plus the lambda-weighted consistency and
        # progress terms when configured. All point the same way (higher = better)
        # so the winner is the argmax of the sum; with no extra term it stays the
        # exec-only argmin penalty.
        combined_env = self.exec_lambda * exec_score_env
        combined = False

        # Consistency arm: only when an IDM is loaded and the policy supplied the
        # decoded dreams.
        dreams = context.get("dream_input")
        if self.cons_scorer is not None and dreams is not None:
            cons = self.cons_scorer.score(env_actions, dreams)
            cons_pen_env = torch.as_tensor(
                cons["cons_penalty_per_env"], dtype=torch.float32
            )
            cons_score_env = self._bounded(cons_pen_env)
            combined_env = combined_env + self.cons_lambda * cons_score_env
            combined = True
            info.update(cons)
            info["cons_score_per_env"] = cons_score_env.tolist()
            if cons_score_env.shape[1] == 1:
                info["cons_score"] = cons_score_env[:, 0].tolist()

        # Progress arm: only when a progress model is loaded and the policy
        # supplied the dreams + conditioning + language. ``r_progress`` is a
        # signed reward (design §10), added directly -- not passed through
        # ``_bounded()``.
        prog_ctx = context.get("progress")
        if self.prog_scorer is not None and prog_ctx is not None:
            prog = self.prog_scorer.score(**prog_ctx)
            prog_score_env = torch.as_tensor(
                prog["progress_reward_per_env"], dtype=torch.float32
            )
            combined_env = combined_env + self.prog_lambda * prog_score_env
            combined = True
            info.update(prog)

        if combined:
            info["combined_score_per_env"] = combined_env.tolist()
            if combined_env.shape[1] == 1:
                info["combined_score"] = combined_env[:, 0].tolist()
            chosen_tensor = torch.argmax(combined_env, dim=0)
        else:
            chosen_tensor = torch.argmin(exec_pen_env, dim=0)

        chosen_per_env = chosen_tensor.cpu().tolist()
        info["chosen_index_per_env"] = chosen_per_env
        info["chosen_counts"] = torch.bincount(
            chosen_tensor.cpu(), minlength=exec_pen_env.shape[0]
        ).tolist()
        chosen = chosen_per_env[0] if len(chosen_per_env) == 1 else chosen_per_env
        info["chosen_index"] = chosen
        return chosen, info
