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


def _load_robometer_progress(path: str, device: str):
    """Load the frozen Robometer reward model for PRM progress scoring (§10).

    In-process load via Robometer's own ``load_model_from_hf`` (NOT a full
    ``pip install -e .`` -- see ``dreamzero_robometer_combined_env_README.md``);
    imports are lazy so this module still imports where Robometer is absent
    (exec-only / non-progress runs). Returns a bundle of the frozen model +
    collator + the discrete-progress decode settings the scorer needs, mirroring
    ``scripts/example_inference_local.py``.
    """
    import torch as _torch

    from robometer.utils.save import load_model_from_hf
    from robometer.utils.setup_utils import setup_batch_collator

    dev = _torch.device(device)
    exp_config, tokenizer, processor, reward_model = load_model_from_hf(
        model_path=str(path), device=dev
    )
    reward_model.eval()
    collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)

    # Discrete-progress decode settings, derived exactly as the reference script
    # (Robometer-4B-LIBERO: progress_discrete_bins=32, discrete loss).
    loss_config = getattr(exp_config, "loss", None)
    is_discrete = (
        getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
        if loss_config
        else False
    )
    num_bins = getattr(loss_config, "progress_discrete_bins", None) or getattr(
        exp_config.model, "progress_discrete_bins", 10
    )
    return {
        "model": reward_model,
        "tokenizer": tokenizer,
        "collator": collator,
        "device": dev,
        "is_discrete": bool(is_discrete),
        "num_bins": int(num_bins),
    }


#: Locked Phase-2 frame contract
#: (``dreamzero_robometer_progress_phase2_input_contract_README.md`` §2): of the
#: F=9 dream frames (0 = conditioning ô_t, 1..8 dreamed) send exactly 4 -- anchor
#: + 3 evenly-spanned futures (env offsets 0/9/18/24) -- so Robometer's collator
#: does no internal subsample and we know which progress_pred is anchor vs future.
ROBOMETER_FRAME_INDICES = (0, 3, 6, 8)
DEFAULT_ROBOMETER_PROGRESS_BATCH_SIZE = 50


class ProgressScorer:
    """Signed progress reward ``r_progress`` for best-of-K candidates (§10).

    Scores each candidate's dreamed future with the frozen **Robometer** reward
    model (``aliangdw/Robometer-4B-LIBERO``) and returns the advance toward task
    completion relative to the dreamed conditioning frame::

        r_progress(c_k, l) = mean(p1, p2, p3) − p0

    where ``[p0, p1, p2, p3]`` are Robometer's per-frame ``progress_pred`` on the
    4 sent frames: the exterior view, hflipped to Robometer orientation, at dream
    indices ``[0, 3, 6, 8]`` (frame 0 = anchor ô_t). Pixels and prompt cross the
    boundary into Robometer's OWN loader / collator / template / forward -- we
    never re-implement a fragment (Phase-2 governing rule). See
    ``dreamzero_robometer_progress_phase2_input_contract_README.md``.

    ``progress_pred`` is in ``[0, 1]`` (vs SmolVLA's ``[-1, 1]``), so
    ``bok_prog_lambda`` must be re-tuned to this range (Phase 6).
    """

    def __init__(
        self,
        bundle: dict,
        frame_indices=ROBOMETER_FRAME_INDICES,
        batch_size: int = DEFAULT_ROBOMETER_PROGRESS_BATCH_SIZE,
    ):
        self.model = bundle["model"]
        self.tokenizer = bundle["tokenizer"]
        self.collator = bundle["collator"]
        self.device = bundle["device"]
        self.is_discrete = bundle["is_discrete"]
        self.num_bins = bundle["num_bins"]
        self.frame_indices = tuple(int(i) for i in frame_indices)
        self.batch_size = max(1, int(batch_size))

    def _candidate_frames(self, dream_rgb: torch.Tensor) -> np.ndarray:
        """``[K,B,2,F,3,H,W]`` dream -> ``[K,B,4,H,W,3]`` uint8 RGB, Robometer-ready.

        Exterior view only (index 0), subsample to the 4 contract frames,
        CHW->HWC, uint8, then hflip the width axis (PI dreams are mirrored vs
        Robometer). Wrist view (index 1) is never sent -- Robometer-4B-LIBERO is
        single-view (agentview).
        """
        dream = torch.as_tensor(dream_rgb, dtype=torch.float32)
        if dream.ndim != 7 or dream.shape[2] != 2:
            raise ValueError(
                f"dream_rgb must be [K, B, 2, F, 3, H, W], got {tuple(dream.shape)}"
            )
        f = dream.shape[3]
        idx = list(self.frame_indices)
        if max(idx) >= f:
            raise ValueError(
                f"frame_indices {idx} exceed F={f} dream frames"
            )
        ext = dream[:, :, 0][:, :, idx]  # [K, B, 4, 3, H, W] exterior, subsampled
        ext = ext.clamp(0, 255).round().to(torch.uint8)
        ext = ext.permute(0, 1, 2, 4, 5, 3).contiguous()  # CHW -> HWC: [K,B,4,H,W,3]
        frames = ext.cpu().numpy()[..., ::-1, :]  # hflip width -> Robometer orient.
        return np.ascontiguousarray(frames)

    @torch.no_grad()
    def _robometer_progress_batch(
        self, frames_batch_hwc: np.ndarray, tasks: list[str]
    ) -> list[np.ndarray]:
        """Batch of clips -> Robometer per-frame ``progress_pred`` outputs.

        Each input clip is a Python-list style multi-image trajectory with the
        locked 4 contract frames. Robometer's own collator, prompt template,
        ``<|prog_token|>`` insertion, resize, and 32-bin softmax-expectation
        still stay inside Robometer.
        """
        from robometer.data.dataset_types import ProgressSample, Trajectory
        from robometer.evals.eval_server import compute_batch_outputs

        samples = []
        for i, frames_hwc in enumerate(frames_batch_hwc):
            task = tasks[i] if i < len(tasks) else ""
            traj = Trajectory(
                frames=frames_hwc,
                frames_shape=tuple(frames_hwc.shape),
                task=task,
                id=str(i),
                metadata={"subsequence_length": int(frames_hwc.shape[0])},
                video_embeddings=None,
            )
            samples.append(ProgressSample(trajectory=traj, sample_type="progress"))

        batch = self.collator(samples)
        progress_inputs = batch["progress_inputs"]
        for key, value in progress_inputs.items():
            if hasattr(value, "to"):
                progress_inputs[key] = value.to(self.device)
        results = compute_batch_outputs(
            self.model,
            self.tokenizer,
            progress_inputs,
            sample_type="progress",
            is_discrete_mode=self.is_discrete,
            num_bins=self.num_bins,
        )
        preds = results.get("progress_pred", [])
        if len(preds) != len(samples):
            raise ValueError(
                f"Robometer returned {len(preds)} progress predictions for "
                f"{len(samples)} clips"
            )
        return [np.asarray(p, dtype=np.float32).ravel() for p in preds]

    @torch.no_grad()
    def _robometer_progress(self, frames_hwc: np.ndarray, task: str) -> np.ndarray:
        """One clip -> Robometer per-frame ``progress_pred`` via their own forward."""
        return self._robometer_progress_batch(
            np.expand_dims(frames_hwc, axis=0), [task]
        )[0]

    @torch.no_grad()
    def score(self, dream_rgb: torch.Tensor, language: list) -> dict[str, Any]:
        """Per-candidate signed progress reward.

        Args:
            dream_rgb: ``[K, B, 2, F, 3, H, W]`` per-candidate decoded dream split
                into exterior/wrist views, pixels in ``[0, 255]``. Frame 0 is the
                conditioning obs ``ô_t``; frames are subsampled to the contract
                set ``[0, 3, 6, 8]`` (anchor + 3 futures).
            language: list of ``B`` verbatim LIBERO instruction strings; element
                ``b`` goes straight to Robometer's prompt template.

        Returns:
            ``progress_reward_per_env`` ``[K, B]`` (signed) plus
            ``progress_future_per_env`` and ``progress_cond_per_env`` for logging;
            flat per-candidate lists for ``B == 1``.
        """
        frames = self._candidate_frames(dream_rgb)  # [K, B, 4, H, W, 3] uint8
        k, b, n_sent = frames.shape[0], frames.shape[1], frames.shape[2]
        lang = list(language)
        flat_frames = frames.reshape(k * b, n_sent, *frames.shape[3:])
        flat_tasks = [
            lang[bi] if bi < len(lang) else "" for _ki in range(k) for bi in range(b)
        ]
        flat_preds = np.zeros((k * b, n_sent), dtype=np.float32)

        for start in range(0, k * b, self.batch_size):
            end = min(start + self.batch_size, k * b)
            batch_preds = self._robometer_progress_batch(
                flat_frames[start:end], flat_tasks[start:end]
            )
            for local_i, p in enumerate(batch_preds):
                if p.shape[0] != n_sent:
                    flat_i = start + local_i
                    ki, bi = divmod(flat_i, b)
                    raise ValueError(
                        f"Robometer returned {p.shape[0]} progress values for "
                        f"{n_sent} sent frames (candidate k={ki}, b={bi}); the "
                        "collator subsampled -- check the frame contract (§2)."
                    )
                flat_preds[start + local_i] = p

        preds = flat_preds.reshape(k, b, n_sent)

        # r_progress = mean(future) − anchor; signed, added directly (not through
        # _bounded). progress_pred ∈ [0, 1], so reward ∈ [-1, 1].
        anchor = torch.as_tensor(preds[..., 0], dtype=torch.float32)  # [K, B]
        future = torch.as_tensor(
            preds[..., 1:].mean(axis=-1), dtype=torch.float32
        )  # [K, B]
        reward = future - anchor  # [K, B]
        out = {"progress_cond_per_env": anchor.tolist()}
        out.update(_per_env_payload("progress_future", future))
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
    dreams + language in ``context["progress"]``, the signed progress reward
    (:class:`ProgressScorer`, Robometer, design §10) is added too. The
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
    ``bok_cons_grip_w``; and for progress ``bok_progress_model_path``
    (Robometer-4B-LIBERO checkpoint dir), ``bok_progress_device``,
    ``bok_prog_lambda``, ``bok_progress_batch_size``.
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
                _load_robometer_progress(
                    str(prog_path),
                    str(getattr(config, "bok_progress_device", "cuda")),
                ),
                batch_size=int(
                    getattr(
                        config,
                        "bok_progress_batch_size",
                        DEFAULT_ROBOMETER_PROGRESS_BATCH_SIZE,
                    )
                ),
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
