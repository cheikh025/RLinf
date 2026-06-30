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
    """Load the frozen Robometer reward model for PRM progress scoring.

    In-process load via Robometer's own ``load_model_from_hf`` (NOT a full
    ``pip install -e .`` -- see ``dreamzero_robometer_combined_env_README.md``);
    imports are lazy so this module still imports where Robometer is absent
    (exec-only / non-progress runs). Returns a bundle of the frozen model +
    collator + the settings the canonical last-frame chain needs:
    ``max_frames`` (read from the checkpoint's ``data.max_frames``) and
    ``model_type`` (``model.model_type``), plus the discrete-progress decode
    settings -- so ``raw_dict_to_sample`` / ``process_batch_helper`` /
    ``extract_rewards_from_output`` run exactly as in the LIBERO reward wrapper.
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

    # Frame budget + model type for Robometer's own inference chain. ``max_frames``
    # comes straight from the checkpoint (for example, 4 for
    # aliangdw/Robometer-4B-LIBERO and 8 for robometer/Robometer-4B) -- no literal
    # here. ``model_type`` is required by ``process_batch_helper``.
    data_config = getattr(exp_config, "data", None)
    max_frames = int(getattr(data_config, "max_frames", 8))
    model_type = getattr(getattr(exp_config, "model", None), "model_type", None)
    if model_type is None:
        raise ValueError(
            "Robometer exp_config.model.model_type is required for progress scoring"
        )
    return {
        "model": reward_model,
        "tokenizer": tokenizer,
        "collator": collator,
        "device": dev,
        "is_discrete": bool(is_discrete),
        "num_bins": int(num_bins),
        "max_frames": max_frames,
        "model_type": model_type,
    }


#: The decoded dream is F=9 frames: index 0 is the conditioning frame ô_t (the
#: real current observation, NOT dreamed) and indices 1..8 are the 8 dreamed
#: future frames. We score the 8 dreamed frames as the candidate trajectory and
#: hand all 8 to Robometer. Robometer then applies the checkpoint's own
#: ``data.max_frames`` setting (LIBERO is 4, so it subsamples internally). The
#: last Robometer progress prediction is the candidate value.
DREAMED_FRAME_START = 1
DREAMED_FRAME_COUNT = 8
DEFAULT_ROBOMETER_PROGRESS_BATCH_SIZE = 50


PRM_TERM_ALIASES = {
    "exec": "exec",
    "executability": "exec",
    "cons": "consistency",
    "consistency": "consistency",
    "idm": "consistency",
    "progress": "progress",
    "prog": "progress",
    "robometer": "progress",
}
PRM_TERM_ORDER = ("exec", "consistency", "progress")


def _config_value_is_set(value: Any) -> bool:
    return value is not None and str(value).strip().lower() not in ("", "none", "null")


def _split_prm_terms(raw_terms: Any) -> list[str]:
    if isinstance(raw_terms, str):
        terms = raw_terms.strip()
        if terms.startswith("[") and terms.endswith("]"):
            terms = terms[1:-1]
        return [
            item.strip()
            for item in terms.replace(";", ",").split(",")
            if item.strip()
        ]
    try:
        return [str(item).strip() for item in raw_terms if str(item).strip()]
    except TypeError:
        return [str(raw_terms).strip()]


def _auto_prm_terms(config: Any) -> tuple[str, ...]:
    terms = ["exec"]
    if _config_value_is_set(getattr(config, "bok_idm_model_path", None)):
        terms.append("consistency")
    if _config_value_is_set(getattr(config, "bok_progress_model_path", None)):
        terms.append("progress")
    return tuple(terms)


def _normalize_prm_terms(config: Any) -> tuple[str, ...]:
    raw_terms = getattr(config, "bok_prm_terms", None)
    if raw_terms is None:
        return _auto_prm_terms(config)

    if isinstance(raw_terms, str) and raw_terms.strip().lower() == "auto":
        return _auto_prm_terms(config)

    requested = _split_prm_terms(raw_terms)
    if len(requested) == 1 and requested[0].lower() == "all":
        requested = list(PRM_TERM_ORDER)

    terms = []
    for term in requested:
        normalized = PRM_TERM_ALIASES.get(term.lower())
        if normalized is None:
            valid = ", ".join(PRM_TERM_ORDER)
            raise ValueError(
                f"Unknown bok_prm_terms entry {term!r}; expected one of: {valid}."
            )
        if normalized not in terms:
            terms.append(normalized)

    if not terms:
        raise ValueError("bok_prm_terms must contain at least one PRM term.")
    return tuple(term for term in PRM_TERM_ORDER if term in terms)


class ProgressScorer:
    """Last-frame progress reward ``r_progress`` for best-of-K candidates.

    Scores each candidate's dreamed future with the frozen **Robometer** reward
    model (``aliangdw/Robometer-4B-LIBERO``) through Robometer's OWN canonical
    inference chain -- exactly the LIBERO reward-wrapper path::

        raw_dict_to_sample(frames, max_frames)            # Robometer linspace-subsample
            -> process_batch_helper(use_frame_steps=False)   # one forward, no 4-frame steps
                -> extract_rewards_from_output            # last-frame progress, clamp [0, 1]

    The candidate value is the last-frame progress of its 8 dreamed frames (dream
    indices 1..8; index 0 is the conditioning ô_t and is not scored). All K
    candidates share ô_t, so the last-frame value ranks them directly. Frames,
    ``max_frames`` and the prompt cross into Robometer's OWN loader / collator /
    template / forward -- we never re-implement a fragment.

    ``progress_pred`` (and thus the reward) is in ``[0, 1]``.
    """

    def __init__(
        self,
        bundle: dict,
        batch_size: int = DEFAULT_ROBOMETER_PROGRESS_BATCH_SIZE,
    ):
        self.model = bundle["model"]
        self.tokenizer = bundle["tokenizer"]
        self.collator = bundle["collator"]
        self.device = bundle["device"]
        self.is_discrete = bundle["is_discrete"]
        self.num_bins = bundle["num_bins"]
        self.max_frames = int(bundle["max_frames"])
        self.model_type = bundle["model_type"]
        self.batch_size = max(1, int(batch_size))

    def _candidate_frames(self, dream_rgb: torch.Tensor) -> np.ndarray:
        """``[K,B,2,F,3,H,W]`` dream -> ``[K,B,8,H,W,3]`` uint8 RGB, Robometer-ready.

        Exterior view only (view index 0), the 8 DREAMED frames (dream indices
        1..8; index 0 is the conditioning ô_t, not scored), CHW->HWC, uint8, then
        hflip the width axis (PI dreams are mirrored vs Robometer). Wrist view
        (view index 1) is never sent -- Robometer-4B-LIBERO is single-view
        (agentview).
        """
        dream = torch.as_tensor(dream_rgb, dtype=torch.float32)
        if dream.ndim != 7 or dream.shape[2] != 2:
            raise ValueError(
                f"dream_rgb must be [K, B, 2, F, 3, H, W], got {tuple(dream.shape)}"
            )
        f = dream.shape[3]
        need = DREAMED_FRAME_START + DREAMED_FRAME_COUNT
        if f < need:
            raise ValueError(
                f"expected F>={need} dream frames (1 conditioning + "
                f"{DREAMED_FRAME_COUNT} dreamed), got F={f}"
            )
        sl = slice(DREAMED_FRAME_START, DREAMED_FRAME_START + DREAMED_FRAME_COUNT)
        ext = dream[:, :, 0][:, :, sl]  # [K, B, 8, 3, H, W] exterior, dreamed
        ext = ext.clamp(0, 255).round().to(torch.uint8)
        # CHW -> HWC: [K, B, 8, H, W, 3]
        ext = ext.permute(0, 1, 2, 4, 5, 3).contiguous()
        frames = ext.cpu().numpy()[..., ::-1, :]  # hflip width -> Robometer orient.
        return np.ascontiguousarray(frames)

    @torch.no_grad()
    def _last_frame_reward_batch(
        self, frames_batch_hwc: np.ndarray, tasks: list[str]
    ) -> list[float]:
        """Batch of dreamed clips -> Robometer last-frame reward, via THEIR code.

        Runs Robometer's reference inference chain (the LIBERO reward wrapper):
        ``raw_dict_to_sample`` linspace-subsamples to the checkpoint's
        ``max_frames`` (for LIBERO, 8 dreamed frames become 4 model frames),
        ``process_batch_helper(use_frame_steps=False)`` does one forward (no
        prefix expansion), and ``extract_rewards_from_output`` returns the
        last-frame progress clamped to ``[0, 1]``. Robometer's collator / prompt
        template / ``<|prog_token|>`` insertion / resize / softmax-expectation all
        stay inside Robometer.
        """
        from robometer.evals.eval_server import process_batch_helper
        from robometer.evals.eval_utils import (
            extract_rewards_from_output,
            raw_dict_to_sample,
        )

        samples = []
        for i, frames_hwc in enumerate(frames_batch_hwc):
            task = tasks[i] if i < len(tasks) else ""
            raw = {
                "frames": np.ascontiguousarray(frames_hwc),
                "task": task,
                "id": str(i),
                "metadata": {"subsequence_length": int(frames_hwc.shape[0])},
                "video_embeddings": None,
                "text_embedding": None,
            }
            samples.append(
                raw_dict_to_sample(
                    raw_data=raw, max_frames=self.max_frames, sample_type="progress"
                )
            )

        outputs = process_batch_helper(
            model_type=self.model_type,
            model=self.model,
            tokenizer=self.tokenizer,
            batch_collator=self.collator,
            device=self.device,
            batch_data=[s.model_dump() for s in samples],
            job_id=0,
            is_discrete_mode=self.is_discrete,
            num_bins=self.num_bins,
            use_frame_steps=False,
        )
        rewards = extract_rewards_from_output(outputs)
        if len(rewards) != len(samples):
            raise ValueError(
                f"Robometer returned {len(rewards)} rewards for {len(samples)} clips"
            )
        return [float(r) for r in rewards]

    @torch.no_grad()
    def score(self, dream_rgb: torch.Tensor, language: list) -> dict[str, Any]:
        """Per-candidate last-frame progress reward.

        Args:
            dream_rgb: ``[K, B, 2, F=9, 3, H, W]`` per-candidate decoded dream
                split into exterior/wrist views, pixels in ``[0, 255]``. Frame 0
                is the conditioning obs ``ô_t``; frames 1..8 are the dreamed future
                and form the scored trajectory.
            language: list of ``B`` verbatim LIBERO instruction strings; element
                ``b`` goes straight to Robometer's prompt template.

        Returns:
            ``progress_reward_per_env`` ``[K, B]`` -- the last-frame progress of
            each candidate's dreamed clip, clamped ``[0, 1]``; flat per-candidate
            list for ``B == 1``.
        """
        frames = self._candidate_frames(dream_rgb)  # [K, B, 8, H, W, 3] uint8
        k, b, n_sent = frames.shape[0], frames.shape[1], frames.shape[2]
        lang = list(language)
        flat_frames = frames.reshape(k * b, n_sent, *frames.shape[3:])
        flat_tasks = [
            lang[bi] if bi < len(lang) else "" for _ki in range(k) for bi in range(b)
        ]
        flat_rewards = np.zeros((k * b,), dtype=np.float32)

        for start in range(0, k * b, self.batch_size):
            end = min(start + self.batch_size, k * b)
            flat_rewards[start:end] = np.asarray(
                self._last_frame_reward_batch(
                    flat_frames[start:end], flat_tasks[start:end]
                ),
                dtype=np.float32,
            )

        reward = torch.as_tensor(flat_rewards.reshape(k, b), dtype=torch.float32)
        return _per_env_payload("progress_reward", reward)


class DreamZeroPRM:
    """Combine PRM terms and select which best-of-K candidate to execute.

    Active terms are controlled by ``bok_prm_terms``. Supported terms are
    ``exec``, ``consistency``, and ``progress``; aliases include ``cons``/``idm``
    and ``prog``/``robometer``. When the key is omitted or set to ``auto``, the
    legacy behavior is preserved: ``exec`` is active, and ``consistency`` /
    ``progress`` are added when their checkpoint paths are configured.

    Selection is always ``argmax`` over the weighted sum of the active terms.
    Candidate 0 has no special tie-break privilege in best-of-K mode.

    Config (read via ``getattr`` from the policy's ``DreamZeroConfig``, all
    optional Hydra ``+actor.model.*`` keys): ``bok_prm_terms``,
    ``bok_exec_w_alim``,
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
        self.active_terms = _normalize_prm_terms(config)
        self.uses_exec = "exec" in self.active_terms
        self.uses_consistency = "consistency" in self.active_terms
        self.uses_progress = "progress" in self.active_terms
        self.exec_lambda = float(getattr(config, "bok_exec_lambda", 1.0))
        self.cons_lambda = float(getattr(config, "bok_cons_lambda", 1.0))

        self.exec_scorer = None
        if self.uses_exec:
            self.exec_scorer = ExecutabilityScorer(
                w_alim=float(getattr(config, "bok_exec_w_alim", 1.0)),
                w_grip=float(getattr(config, "bok_exec_w_grip", 1.0)),
                w_acc=float(getattr(config, "bok_exec_w_acc", 0.1)),
                w_jerk=float(getattr(config, "bok_exec_w_jerk", 0.05)),
            )

        # Optional cycle-consistency term. Built only when requested by
        # ``bok_prm_terms``.
        self.cons_scorer = None
        self.cons_uses_latent = False
        idm_path = getattr(config, "bok_idm_model_path", None)
        if self.uses_consistency:
            if not _config_value_is_set(idm_path):
                raise ValueError(
                    "bok_prm_terms includes 'consistency' but "
                    "bok_idm_model_path is not set."
                )
            idm, self.cons_uses_latent = _load_consistency_idm(
                str(idm_path),
                str(getattr(config, "bok_idm_device", "cuda")),
            )
            self.cons_scorer = ConsistencyScorer(
                idm,
                w_arm=float(getattr(config, "bok_cons_arm_w", 1.0)),
                w_grip=float(getattr(config, "bok_cons_grip_w", 1.0)),
            )

        # Optional progress reward term. Built only when requested by
        # ``bok_prm_terms``.
        self.prog_scorer = None
        self.prog_lambda = float(getattr(config, "bok_prog_lambda", 1.0))
        prog_path = getattr(config, "bok_progress_model_path", None)
        if self.uses_progress:
            if not _config_value_is_set(prog_path):
                raise ValueError(
                    "bok_prm_terms includes 'progress' but "
                    "bok_progress_model_path is not set."
                )
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

    @staticmethod
    def _add_term(
        combined: Optional[torch.Tensor],
        contribution: torch.Tensor,
    ) -> torch.Tensor:
        return contribution if combined is None else combined + contribution

    def select(
        self,
        env_actions: torch.Tensor,
        context: Optional[dict] = None,
    ) -> tuple[Any, dict]:
        """Pick the candidate to execute.

        Args:
            env_actions: ``[K, B, T, D]`` env-space candidate action chunks.
            context: optional extras: ``prev_action`` for ``exec``,
                ``dream_input`` for ``consistency``, and ``progress`` for
                Robometer progress.

        Returns:
            ``(chosen_index, info)`` where ``chosen_index`` is an ``int`` for
            ``B == 1`` and a list of length ``B`` for parallel eval. ``info``
            holds per-env matrices and selection metadata for logging. Flat
            candidate lists are included only for ``B == 1``.
        """
        context = context or {}
        action_tensor = torch.as_tensor(env_actions, dtype=torch.float32)
        if action_tensor.ndim != 4:
            raise ValueError(
                f"env_actions must be [K, B, T, D], got {tuple(action_tensor.shape)}"
            )
        k = int(action_tensor.shape[0])
        info = {"prm_terms": list(self.active_terms)}
        combined_env = None

        if self.uses_exec:
            terms = self.exec_scorer.score(
                action_tensor, prev_action=context.get("prev_action")
            )
            exec_pen_env = torch.as_tensor(
                terms["penalty_per_env"], dtype=torch.float32
            )
            exec_score_env = self._bounded(exec_pen_env)
            info.update(terms)
            info["exec_score_per_env"] = exec_score_env.tolist()
            if exec_score_env.shape[1] == 1:
                info["score"] = exec_score_env[:, 0].tolist()
                info["exec_penalty"] = terms["penalty"]
                info["exec_score"] = info["score"]
            combined_env = self._add_term(
                combined_env, self.exec_lambda * exec_score_env
            )

        if self.uses_consistency:
            dreams = context.get("dream_input")
            if dreams is None:
                raise ValueError(
                    "PRM term 'consistency' is active, but context['dream_input'] "
                    "was not supplied."
                )
            cons = self.cons_scorer.score(env_actions, dreams)
            cons_pen_env = torch.as_tensor(
                cons["cons_penalty_per_env"], dtype=torch.float32
            )
            cons_score_env = self._bounded(cons_pen_env)
            combined_env = self._add_term(
                combined_env, self.cons_lambda * cons_score_env
            )
            info.update(cons)
            info["cons_score_per_env"] = cons_score_env.tolist()
            if cons_score_env.shape[1] == 1:
                info["cons_score"] = cons_score_env[:, 0].tolist()

        if self.uses_progress:
            prog_ctx = context.get("progress")
            if prog_ctx is None:
                raise ValueError(
                    "PRM term 'progress' is active, but context['progress'] "
                    "was not supplied."
                )
            prog = self.prog_scorer.score(**prog_ctx)
            prog_score_env = torch.as_tensor(
                prog["progress_reward_per_env"], dtype=torch.float32
            )
            combined_env = self._add_term(
                combined_env, self.prog_lambda * prog_score_env
            )
            info.update(prog)

        if combined_env is None:
            raise ValueError("No active PRM terms were scored.")

        info["combined_score_per_env"] = combined_env.tolist()
        if combined_env.shape[1] == 1:
            info["combined_score"] = combined_env[:, 0].tolist()
        chosen_tensor = torch.argmax(combined_env, dim=0)

        chosen_per_env = chosen_tensor.cpu().tolist()
        info["chosen_index_per_env"] = chosen_per_env
        info["chosen_counts"] = torch.bincount(
            chosen_tensor.cpu(), minlength=k
        ).tolist()
        chosen = chosen_per_env[0] if len(chosen_per_env) == 1 else chosen_per_env
        info["chosen_index"] = chosen
        return chosen, info
