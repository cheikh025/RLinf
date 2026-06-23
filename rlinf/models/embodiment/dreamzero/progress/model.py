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

"""SmolVLA-based frame-level progress value model ``Vθ(o, l) ∈ [-1, 1]``.

Milestone 1 of the DreamZero progress model (design record:
``dreamzero_docs/dreamzero_progress_model_design_README.md``). The model scores
whether a visual state advances a LIBERO language instruction, adding a
semantic task-progress signal to the existing executability and IDM-consistency
PRM terms. It is a *state*-value model: it consumes images and language only,
never candidate actions (§5 of the design).

The value head matches RISE's training code (``OpenDriveLab/RISE``,
``openpi_value/models_pytorch/pi0_pytorch.py``, ``exist_negative_progress``
mode): a three-layer MLP read off the action-expert suffix, bounded to
``[-1, 1]`` by a final ``tanh`` (RISE disables BCE in this mode "due to the
presence of negative progress labels"). It is trained with the ``t / T``
progress target on successes plus success/failure TD learning with an EMA
target. Only the vision-language **backbone differs** from RISE (SmolVLA's
SmolVLM2 here vs. pi-0.5); everything else is kept as close to RISE as possible.

Architecture (§3):

- load the official SmolVLA base checkpoint ``lerobot/smolvla_base``;
- keep the complete **SmolVLM2** vision-language backbone frozen and in eval
  mode for the full run (gradients stop at the VLM output);
- retain and fine-tune the pretrained **SmolVLA action expert**;
- replace the action-token input / action-output projection with one learned
  ``[VALUE]`` query token and a three-layer scalar ``tanh`` value head;
- drop ``state_proj`` / ``action_in_proj`` / ``action_time_mlp_*`` /
  ``action_out_proj``, flow-matching noise, and diffusion timesteps entirely.

The forward mirrors :meth:`VLAFlowMatching.forward`'s prefix/suffix protocol
exactly, but the suffix is a single value-query token (not a noised action
chunk) and the prefix carries **no state token** -- so the model reuses the
checkpoint's verbatim multi-view image/language token formatting and the
expert's pretrained attention, with no robot-state pathway.

Loss functions (RISE progress regression + TD) and the training schedule are
Milestone 3 and live in ``progress/train.py``; the dataset is Milestone 2 in
``progress/data.py``.
"""

import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn

from rlinf.utils.logging import get_logger

logger = get_logger()

#: Projections in :class:`VLAFlowMatching` the progress model never uses; they
#: are deleted after load so the expert sees only images, language, and the
#: value query (design §3: no state, no action, no diffusion-time input).
_UNUSED_VLA_PROJECTIONS = (
    "state_proj",
    "action_in_proj",
    "action_out_proj",
    "action_time_mlp_in",
    "action_time_mlp_out",
)


@dataclass
class ProgressModelConfig:
    """Configuration for :class:`SmolVLAProgressModel`.

    The value is bounded to ``[-1, 1]`` by a final ``tanh``, matching RISE's
    ``pi0_pytorch.py`` value head in ``exist_negative_progress`` mode (negative
    values mark failure states). ``value_head_hidden`` defaults to the expert
    hidden size.
    """

    smolvla_checkpoint: str = "lerobot/smolvla_base"
    value_query_std: float = 0.02
    value_head_hidden: Optional[int] = None


def build_value_head(d_in: int, hidden: int) -> nn.Sequential:
    """Three-layer scalar value head (RISE ``pi0_pytorch`` head, design §3.1).

    ``Linear(d, h) -> SiLU -> Linear(h, h) -> SiLU -> Linear(h, 1) -> Tanh``
    (SiLU == RISE's swish), bounding the value to ``[-1, 1]``. Factored out, and
    free of SmolVLA, so it is testable without the checkpoint.
    """
    return nn.Sequential(
        nn.Linear(d_in, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, 1),
        nn.Tanh(),
    )


def make_value_query(d_expert: int, std: float = 0.02) -> nn.Parameter:
    """One learned ``[VALUE]`` query token ``∈ R^(1 x 1 x d_expert)`` (design §3.1).

    Initialized from ``N(0, std)`` and passed to the action expert as a
    one-token suffix that cross-attends to the frozen image/language
    representation. Factored out for unit testing without the checkpoint.
    """
    query = nn.Parameter(torch.empty(1, 1, d_expert))
    nn.init.normal_(query, mean=0.0, std=float(std))
    return query


class SmolVLAProgressModel(nn.Module):
    """``Vθ(o, l) ∈ [-1, 1]`` built on a frozen SmolVLM2 + trainable SmolVLA expert.

    Wraps a loaded :class:`SmolVLAPolicy`: the policy's ``model.vlm_with_expert``
    supplies the frozen vision-language backbone and the pretrained action
    expert; this module adds the value query and value head and exposes a
    progress-only forward.

    Parameter policy (design §3.2): SmolVLM2 is always frozen and in eval mode;
    the value query and value head train from step 0; the action expert is
    frozen until the trainer calls :meth:`unfreeze_expert` (Stage 2). The expert
    is also held in **eval mode throughout** (dropout off) so the train-time and
    eval/inference forwards match -- it still trains (gradients flow in eval
    mode), but its dropout never perturbs the features the value head reads,
    which otherwise makes the eval-mode value collapse toward -1. Use
    :meth:`param_groups` for the two-learning-rate optimizer.
    """

    def __init__(self, config: Optional[ProgressModelConfig] = None, policy=None):
        super().__init__()
        self.config = config or ProgressModelConfig()

        # Lazy import: this module must stay importable without lerobot (e.g.
        # for unit-testing build_value_head / make_value_query). Only building
        # the full model pulls SmolVLA in.
        if policy is None:
            policy = self._load_smolvla_policy(self.config.smolvla_checkpoint)
        self.policy = policy
        self.vla = policy.model  # VLAFlowMatching
        self.vlm_with_expert = self.vla.vlm_with_expert

        # Drop the action/state/diffusion-time projections: the progress model
        # never feeds state, actions, or a timestep, so these stay unused and
        # out of the checkpoint (design §3).
        for name in _UNUSED_VLA_PROJECTIONS:
            if hasattr(self.vla, name):
                delattr(self.vla, name)

        d_expert = int(self.vlm_with_expert.expert_hidden_size)
        hidden = self.config.value_head_hidden or d_expert
        self.value_query = make_value_query(d_expert, self.config.value_query_std)
        self.value_head = build_value_head(d_expert, hidden)

        # SmolVLM2 frozen + eval for the whole run; expert frozen until Stage 2
        # but always held in eval (dropout off) so train/eval forwards match.
        self.freeze_vlm()
        self.freeze_expert()
        self.vlm_with_expert.lm_expert.eval()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_smolvla_policy(checkpoint: str):
        """Load ``SmolVLAPolicy.from_pretrained(checkpoint)`` (lazy import)."""
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        logger.info("Loading SmolVLA base policy from %s", checkpoint)
        return SmolVLAPolicy.from_pretrained(checkpoint)

    # ------------------------------------------------------------------
    # Freeze / train-mode control (design §3.2)
    # ------------------------------------------------------------------

    def freeze_vlm(self) -> None:
        """Freeze the complete SmolVLM2 backbone and hold it in eval mode."""
        self.vlm_with_expert.vlm.eval()
        for p in self.vlm_with_expert.vlm.parameters():
            p.requires_grad_(False)

    def freeze_expert(self) -> None:
        """Freeze the SmolVLA action expert (Stage 1 warm-up)."""
        for p in self.vlm_with_expert.lm_expert.parameters():
            p.requires_grad_(False)
        self._expert_trainable = False

    def unfreeze_expert(self) -> None:
        """Make the action expert trainable (Stage 2, step 2,000 onward)."""
        for p in self.vlm_with_expert.lm_expert.parameters():
            p.requires_grad_(True)
        self._expert_trainable = True

    @property
    def expert_trainable(self) -> bool:
        return bool(getattr(self, "_expert_trainable", False))

    def train(self, mode: bool = True) -> "SmolVLAProgressModel":
        """Set train mode but hold SmolVLM2 *and* the action expert in eval.

        The backbone is frozen, and the expert -- trainable from Stage 2 -- is
        kept in eval mode so its dropout is **off in both training and
        eval/inference**. Gradients still flow through an eval-mode module, so
        the expert trains normally; eval only disables dropout. Without this the
        value head learns against the expert's train-mode (dropout-on) features
        and then collapses toward -1 at eval -- where the model is actually
        deployed (design §3.2).
        """
        super().train(mode)
        self.vlm_with_expert.vlm.eval()
        self.vlm_with_expert.lm_expert.eval()
        return self

    def expert_parameters(self):
        return self.vlm_with_expert.lm_expert.parameters()

    def value_parameters(self):
        return [self.value_query, *self.value_head.parameters()]

    def param_groups(self, value_lr: float, expert_lr: float) -> list[dict]:
        """Two optimizer groups: new value modules vs the action expert (§7.1)."""
        return [
            {"params": self.value_parameters(), "lr": float(value_lr)},
            {"params": list(self.expert_parameters()), "lr": float(expert_lr)},
        ]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _embed_prefix(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Image + language prefix, mirroring :meth:`VLAFlowMatching.embed_prefix`
        verbatim **minus the state token** (design §3: no state input).

        Reuses the loaded model's own multi-view formatting (image special
        tokens, SigLIP embedding via ``embed_image``, language embedding and
        ``sqrt(d)`` normalization), so the token layout matches the checkpoint
        for whatever lerobot version is installed.
        """
        from lerobot.policies.smolvla.modeling_smolvla import pad_tensor

        vla = self.vla
        vlm = self.vlm_with_expert
        embs: list[torch.Tensor] = []
        pad_masks: list[torch.Tensor] = []
        att_masks: list[int] = []

        for img, img_mask in zip(images, img_masks, strict=False):
            if vla.add_image_special_tokens:
                start_tok = (
                    vlm.embed_language_tokens(
                        vla.global_image_start_token.to(device=vlm.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                start_mask = torch.ones_like(
                    start_tok[:, :, 0], dtype=torch.bool, device=start_tok.device
                )
                embs.append(start_tok)
                pad_masks.append(start_mask)
                att_masks += [0] * start_mask.shape[-1]

            img_emb = vlm.embed_image(img)
            img_emb = img_emb * torch.tensor(
                img_emb.shape[-1] ** 0.5, dtype=img_emb.dtype, device=img_emb.device
            )
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

            if vla.add_image_special_tokens:
                end_tok = (
                    vlm.embed_language_tokens(
                        vla.image_end_token.to(device=vlm.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                end_mask = torch.ones_like(
                    end_tok[:, :, 0], dtype=torch.bool, device=end_tok.device
                )
                embs.append(end_tok)
                pad_masks.append(end_mask)
                att_masks += [0] * end_mask.shape[1]

        lang_emb = vlm.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * (lang_emb.shape[-1] ** 0.5)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        # Pad to the configured prefix length exactly as the action path does,
        # so position ids over [prefix, value] line up with the checkpoint.
        seq_len = pad_masks.shape[1]
        if seq_len < vla.prefix_length:
            embs = pad_tensor(embs, vla.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, vla.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, vla.prefix_length, pad_value=0)

        att_masks = att_masks.expand(pad_masks.shape[0], -1)
        return embs, pad_masks, att_masks

    def _value_suffix(
        self, bsize: int, device, dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One value-query token as the expert suffix (mirrors ``embed_suffix``).

        ``att_mask = 1`` marks a new attention block so the prefix cannot attend
        to the value token while the value token attends over all of the
        prefix -- the same boundary the action tokens get.
        """
        suffix = self.value_query.to(device=device, dtype=dtype).expand(bsize, -1, -1)
        pad = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        att = torch.ones(bsize, 1, dtype=dtype, device=device)
        return suffix, pad, att

    def forward(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Predict ``Vθ(o, l) ∈ [-1, 1]`` for a batch of frames.

        Args:
            images: list of ``[B, 3, H, W]`` view tensors already preprocessed
                by the SmolVLA image path (resize-with-pad, range ``[-1, 1]``);
                e.g. ``[exterior, wrist]``.
            img_masks: list of ``[B]`` bool masks, one per view (present cameras).
            lang_tokens: ``[B, L]`` tokenized verbatim LIBERO instruction.
            lang_masks: ``[B, L]`` bool attention mask for the language tokens.

        Returns:
            ``[B]`` float32 task value in ``[-1, 1]`` (tanh-bounded).
        """
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        prefix_embs, prefix_pad, prefix_att = self._embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        bsize = prefix_embs.shape[0]
        suffix_embs, suffix_pad, suffix_att = self._value_suffix(
            bsize, prefix_embs.device, prefix_embs.dtype
        )

        pad_masks = torch.cat([prefix_pad, suffix_pad], dim=1)
        att_masks = torch.cat([prefix_att, suffix_att], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        # The single value token is the last suffix position; upcast like the
        # action path before the head.
        value_token = suffix_out[:, -1].to(dtype=torch.float32)
        return self.value_head(value_token).squeeze(-1)

    # ------------------------------------------------------------------
    # Checkpoint I/O -- save only the trainable parts; reload the frozen
    # SmolVLM2 from the base checkpoint (it never changes), mirroring the
    # IDM's ``from_checkpoint`` convention.
    # ------------------------------------------------------------------

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """State dict of the parts that train: expert + value query + head."""
        sd = {f"expert.{k}": v for k, v in self.vlm_with_expert.lm_expert.state_dict().items()}
        sd["value_query"] = self.value_query.detach()
        for k, v in self.value_head.state_dict().items():
            sd[f"value_head.{k}"] = v
        return sd

    def load_trainable_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        """Inverse of :meth:`trainable_state_dict`."""
        expert_sd = {
            k[len("expert."):]: v for k, v in sd.items() if k.startswith("expert.")
        }
        head_sd = {
            k[len("value_head."):]: v
            for k, v in sd.items()
            if k.startswith("value_head.")
        }
        self.vlm_with_expert.lm_expert.load_state_dict(expert_sd)
        self.value_head.load_state_dict(head_sd)
        with torch.no_grad():
            self.value_query.copy_(sd["value_query"].to(self.value_query.device))

    def save_checkpoint(
        self,
        path: str,
        step: Optional[int] = None,
        stats: Optional[dict] = None,
        thresholds: Optional[dict] = None,
        extra: Optional[dict] = None,
    ) -> None:
        """Write a progress-model checkpoint (trainable parts only).

        Stores the config (so the base SmolVLM2 is reloaded on resume), the
        trainable state dict, optional training ``step`` / dataset ``stats``,
        and the frozen executability/consistency ``thresholds`` the selector
        needs (design §12). A directory writes ``progress.pt``; a file path is
        used as-is.
        """
        if os.path.isdir(path) or not path.endswith(".pt"):
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, "progress.pt")
        torch.save(
            {
                "progress_cfg": asdict(self.config),
                "trainable": self.trainable_state_dict(),
                "step": step,
                "stats": stats,
                "thresholds": thresholds,
                **(extra or {}),
            },
            path,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "SmolVLAProgressModel":
        """Rebuild from a checkpoint dir/file: load base SmolVLA, then the
        trained expert + value modules. Returned frozen and in eval mode.
        """
        ckpt_path = path
        if os.path.isdir(path):
            candidate = os.path.join(path, "progress.pt")
            if not os.path.isfile(candidate):
                raise FileNotFoundError(f"No progress.pt in checkpoint dir {path!r}")
            ckpt_path = candidate
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "progress_cfg" not in ckpt or "trainable" not in ckpt:
            raise KeyError(
                f"Progress checkpoint {ckpt_path!r} missing 'progress_cfg'/'trainable'."
            )
        model = cls(ProgressModelConfig(**ckpt["progress_cfg"]))
        model.load_trainable_state_dict(ckpt["trainable"])
        model.thresholds = ckpt.get("thresholds")
        model.stats = ckpt.get("stats")
        return model.eval().to(device=device, dtype=dtype).requires_grad_(False)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Trainable vs frozen vs total parameter counts (logging helper)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "frozen": total - trainable, "total": total}


def save_config_json(config: ProgressModelConfig, path: str) -> None:
    """Write the model config as JSON (alongside the run output)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)
