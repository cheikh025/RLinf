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

"""RISE training targets for the DreamZero progress value model (design §6).

Mirrors RISE's ``pi0_pytorch.py`` value training (``exist_negative_progress``
mode): two scalar mean-squared-error losses plus the EMA target network, with a
``tanh`` value in ``[-1, 1]``.

- :func:`progress_regression_loss` -- temporal progress regression on
  successful trajectories only, with RISE's target ``y(t) = clip(t / T, -1, 1)``;
- :func:`td_loss` -- success/failure TD learning,
  ``y_TD = r_t + gamma * (1 - done) * V_target(o_{t+1}, l)`` clipped to
  ``[-1, 1]``;
- :class:`EMATarget` -- exponential-moving-average copy of the trainable
  modules (expert + value query + value head) sharing the online model's
  frozen SmolVLM2.

Reward convention (RISE ``pi0_pytorch.py`` lines 632-641): a state is *terminal*
when it falls in the last :data:`TERMINAL_WINDOW` frames
(``|t - T| <= terminal_window``). Terminal states get ``+1`` (success) or
``-1`` (failure); every other state gets ``0``. ``done`` is set on the same
terminal window, so terminal states do not bootstrap.
"""

import copy

import torch

#: TD discount factor (RISE ``value_gamma``).
DEFAULT_GAMMA = 0.995
#: EMA mixing rate applied after every optimizer step (RISE ``value_TD_TAU``).
DEFAULT_TAU = 0.01
#: Reward window: the last N frames of an episode carry the terminal reward
#: (RISE ``value_terminal_window``).
TERMINAL_WINDOW = 10
#: Terminal rewards (RISE ``value_success_reward`` / ``value_failure_reward``).
SUCCESS_REWARD = 1.0
FAILURE_REWARD = -1.0


def progress_target(t_index: torch.Tensor, episode_len: torch.Tensor) -> torch.Tensor:
    """Linear progress label ``y(t) = clip(t / T, -1, 1)`` (RISE eq. 5).

    Matches RISE's ``pi0_pytorch.py``: ``clip(frame_index / episode_length,
    -1, 1)`` in ``exist_negative_progress`` mode. On successful trajectories
    (the only ones this loss scores) ``t / T`` is already in ``[0, 1]``.

    Args:
        t_index: ``[B]`` zero-based frame index within its episode.
        episode_len: ``[B]`` episode length ``T`` (frames).

    Returns:
        ``[B]`` target in ``[-1, 1]``.
    """
    denom = episode_len.to(torch.float32).clamp_min(1.0)
    return (t_index.to(torch.float32) / denom).clamp(-1.0, 1.0)


def progress_regression_loss(
    values: torch.Tensor,
    t_index: torch.Tensor,
    episode_len: torch.Tensor,
    success_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Temporal progress regression over successful frames only (RISE eq. 5).

    ``L_progress = mean_success_frames (Vθ(o_t, l) - t/T)²``. Failed trajectories
    receive no ``t/T`` label (RISE applies progress regression to expert data
    only, ``* is_expert_data``), so they are masked out here.

    Args:
        values: ``[B]`` predicted ``Vθ(o_t, l)`` in ``[-1, 1]``.
        t_index: ``[B]`` zero-based frame index.
        episode_len: ``[B]`` episode length ``T``.
        success_mask: ``[B]`` bool, True for frames from successful episodes.

    Returns:
        ``(loss, metrics)``. ``loss`` is a scalar (0 when no success frames).
    """
    target = progress_target(t_index, episode_len).to(values.dtype)
    mask = success_mask.to(values.dtype)
    n = mask.sum().clamp_min(1.0)
    loss = (((values - target) ** 2) * mask).sum() / n
    metrics = {
        "progress_loss": float(loss.detach()),
        "progress_mae": float((((values - target).abs() * mask).sum() / n).detach()),
        "num_success_frames": int(success_mask.sum().detach()),
    }
    return loss, metrics


def td_reward_and_done(
    t_index: torch.Tensor,
    episode_len: torch.Tensor,
    success_mask: torch.Tensor,
    terminal_window: int = TERMINAL_WINDOW,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-state reward and ``done`` over the terminal window (RISE pi0_pytorch).

    A state is *terminal* when ``|t - T| <= terminal_window`` (the last frames of
    the episode). Terminal success states get ``+1``, terminal failure states
    get ``-1``, every other state gets ``0``; ``done`` marks the same window so
    those states do not bootstrap.
    """
    is_terminal = (
        (t_index.to(torch.float32) - episode_len.to(torch.float32)).abs()
        <= terminal_window
    ).to(torch.float32)
    is_failure = (~success_mask.bool()).to(torch.float32)
    reward = is_terminal * (
        is_failure * FAILURE_REWARD + (1.0 - is_failure) * SUCCESS_REWARD
    )
    return reward, is_terminal


def td_target(
    reward: torch.Tensor,
    done: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float = DEFAULT_GAMMA,
) -> torch.Tensor:
    """Bellman backup clipped to ``[-1, 1]`` (RISE eq. 6).

    ``y = r + gamma * (1 - done) * V_target(o_{t+1})``, so terminal-window states
    (``done == 1``) do not bootstrap. ``next_values`` is the detached EMA target.
    """
    y = reward.to(next_values.dtype) + gamma * (
        1.0 - done.to(next_values.dtype)
    ) * next_values
    return y.clamp(-1.0, 1.0)


def td_loss(
    values: torch.Tensor,
    next_values: torch.Tensor,
    t_index: torch.Tensor,
    episode_len: torch.Tensor,
    success_mask: torch.Tensor,
    gamma: float = DEFAULT_GAMMA,
    terminal_window: int = TERMINAL_WINDOW,
) -> tuple[torch.Tensor, dict]:
    """Success+failure TD learning (RISE eq. 6).

    ``L_TD = mean (Vθ(o_t, l) - stop_gradient(y_TD))²`` over **all** transitions
    (success and failure, unlike progress regression). The windowed reward and
    ``done`` come from :func:`td_reward_and_done`.

    Args:
        values: ``[B]`` predicted ``Vθ(o_t, l)`` (online, with grad).
        next_values: ``[B]`` ``V_target(o_{t+1}, l)`` from the EMA target.
        t_index / episode_len: ``[B]`` frame index and episode length.
        success_mask: ``[B]`` bool, True for successful episodes.
        gamma: discount factor.
        terminal_window: terminal-reward window size.

    Returns:
        ``(loss, metrics)``.
    """
    reward, done = td_reward_and_done(
        t_index, episode_len, success_mask, terminal_window
    )
    y = td_target(reward, done, next_values.detach(), gamma)
    loss = ((values - y) ** 2).mean()
    metrics = {
        "td_loss": float(loss.detach()),
        "td_target_mean": float(y.mean().detach()),
        "num_terminal": int(done.sum().detach()),
    }
    return loss, metrics


class EMATarget:
    """Exponential-moving-average target network (design §6.3).

    Holds EMA copies of the trainable modules -- the SmolVLA action expert, the
    value-query token, and the value head -- while sharing the online model's
    frozen SmolVLM2 weights (so there is no second backbone in memory). Built
    from an online :class:`SmolVLAProgressModel` at step 10,000 and updated
    after every optimizer step with
    ``theta_target <- (1 - tau) * theta_target + tau * theta_online``.

    The target is frozen (no gradients) and always in eval mode; query it with
    :meth:`value` to produce ``V_target(o, l)`` for the TD bootstrap.
    """

    def __init__(self, online):
        from rlinf.models.embodiment.dreamzero.progress.model import (
            SmolVLAProgressModel,
        )

        # Build a target that owns EMA copies of the expert + value modules but
        # *shares* the online frozen SmolVLM2 (no second backbone). Deep-copy the
        # expert wrapper with the VLM temporarily detached so the big backbone is
        # never duplicated, even transiently, then rebind the shared VLM.
        online_vwe = online.vlm_with_expert
        saved_vlm = online_vwe.vlm
        online_vwe.vlm = None
        try:
            target_vwe = copy.deepcopy(online_vwe)  # copies lm_expert; vlm is None
        finally:
            online_vwe.vlm = saved_vlm
        target_vwe.vlm = saved_vlm  # share the frozen VLM

        target = SmolVLAProgressModel.__new__(SmolVLAProgressModel)
        torch.nn.Module.__init__(target)
        target.config = online.config
        target.policy = online.policy
        target.vla = online.vla  # shared prefix-formatting attributes
        target.vlm_with_expert = target_vwe
        target.value_query = torch.nn.Parameter(online.value_query.detach().clone())
        target.value_head = copy.deepcopy(online.value_head)
        target._expert_trainable = False

        self.target = target.eval().requires_grad_(False)
        # Cache the parameter handles paired by key for a cheap in-place lerp.
        self._online_keys = list(online.trainable_state_dict().keys())

    @torch.no_grad()
    def update(self, online, tau: float = DEFAULT_TAU) -> None:
        """In-place EMA step over expert + value query + value head.

        ``trainable_state_dict`` returns storage-sharing views of the live
        parameters (module ``state_dict`` tensors, and ``value_query.detach()``
        which aliases the parameter), so the in-place lerp writes straight into
        the target's parameters.
        """
        online_sd = online.trainable_state_dict()
        target_sd = self.target.trainable_state_dict()
        for k in self._online_keys:
            target_sd[k].mul_(1.0 - tau).add_(
                online_sd[k].to(target_sd[k].device), alpha=tau
            )

    @torch.no_grad()
    def value(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> torch.Tensor:
        """``V_target(o, l)`` for the TD bootstrap (detached, eval mode)."""
        return self.target(images, img_masks, lang_tokens, lang_masks).detach()
