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

"""RISE training entrypoint for the DreamZero progress value model (design §7).

Single-GPU trainer for :class:`...progress.model.SmolVLAProgressModel` on the
combined PI + NVIDIA LIBERO frames (design §8). It runs the fixed three-stage,
50,000-step schedule:

- **Stage 1** (0-1,999): train the value query + head only (SmolVLM2 and the
  action expert frozen). Loss ``L_progress``.
- **Stage 2** (2,000-9,999): unfreeze the action expert. Loss ``L_progress``.
- **Stage 3** (10,000-49,999): add success/failure TD. Loss
  ``L_progress + L_TD``; the EMA target is initialized at step 10,000 and
  updated after every optimizer step.

Optimizer (§7.1): AdamW, bf16 autocast, global batch 64, grad clip 1.0, 1,000
warm-up steps, cosine decay, with **separate parameter groups** -- the value
query/head at 1.0e-4 and the action expert at 2.5e-5. SmolVLM2 stays frozen and
in eval mode throughout (§3.2).

Example::

    python -m rlinf.models.embodiment.dreamzero.progress.train \
        --pi-root /data/physical-intelligence/libero \
        --nvidia-root /data/nvidia/LIBERO-Cosmos-Policy/all_episodes \
        --output ./checkpoints/progress --steps 50000 --batch 64
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rlinf.models.embodiment.dreamzero.idm.train import lr_at, upload_to_huggingface
from rlinf.models.embodiment.dreamzero.progress.data import (
    ProgressTransitionDataset,
    SmolVLAProgressCollator,
    build_progress_episodes,
    grouped_split,
)
from rlinf.models.embodiment.dreamzero.progress.losses import (
    DEFAULT_GAMMA,
    DEFAULT_TAU,
    TERMINAL_WINDOW,
    EMATarget,
    progress_regression_loss,
    td_loss,
)
from rlinf.models.embodiment.dreamzero.progress.model import (
    ProgressModelConfig,
    SmolVLAProgressModel,
    count_parameters,
)

# Schedule boundaries (design §7). Expert unfreezes at Stage 2; TD + EMA begin
# at Stage 3.
STAGE2_STEP = 2_000
STAGE3_STEP = 10_000


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pi-root", default=None, help="physical-intelligence/libero root.")
    p.add_argument(
        "--nvidia-root",
        default=None,
        help="nvidia/LIBERO-Cosmos-Policy all_episodes root (success + failure).",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--smolvla-checkpoint",
        default="lerobot/smolvla_base",
        help="SmolVLA base checkpoint to initialize from.",
    )
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--value-lr", type=float, default=1.0e-4)
    p.add_argument("--expert-lr", type=float, default=2.5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=1_000)
    p.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    p.add_argument("--tau", type=float, default=DEFAULT_TAU)
    p.add_argument("--terminal-window", type=int, default=TERMINAL_WINDOW)
    p.add_argument("--stage2-step", type=int, default=STAGE2_STEP)
    p.add_argument("--stage3-step", type=int, default=STAGE3_STEP)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--val-interval", type=int, default=2_000)
    p.add_argument("--val-batches", type=int, default=50)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--save-interval", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--test-fraction", type=float, default=0.05)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--hf-repo-id", default=None)
    p.add_argument("--hf-token", default=None)
    p.add_argument("--hf-private", action="store_true")
    return p


def _move_views(views: list[torch.Tensor], device: str) -> list[torch.Tensor]:
    return [v.to(device, non_blocking=True) for v in views]


def _batch_to_device(batch: dict, device: str) -> dict:
    out = dict(batch)
    out["images"] = _move_views(batch["images"], device)
    out["img_masks"] = _move_views(batch["img_masks"], device)
    out["next_images"] = _move_views(batch["next_images"], device)
    out["next_img_masks"] = _move_views(batch["next_img_masks"], device)
    for k in ("lang_tokens", "lang_masks", "t_index", "episode_len", "success_mask"):
        out[k] = batch[k].to(device, non_blocking=True)
    return out


@torch.no_grad()
def run_validation(
    model: SmolVLAProgressModel,
    loader: DataLoader,
    device: str,
    max_batches: int,
) -> dict:
    """Offline value dashboard (design §13.1): success-frame progress MAE and
    success/failure terminal-value separation (means + AUROC)."""
    from rlinf.models.embodiment.dreamzero.progress.losses import (
        TERMINAL_WINDOW,
        progress_target,
    )

    model.eval()
    mae_sum, mae_n = 0.0, 0
    succ_term, fail_term = [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = _batch_to_device(batch, device)
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                            dtype=torch.bfloat16):
            v = model(batch["images"], batch["img_masks"],
                      batch["lang_tokens"], batch["lang_masks"]).float()
        succ = batch["success_mask"]
        if succ.any():
            y = progress_target(batch["t_index"], batch["episode_len"])
            mae_sum += float(((v - y).abs() * succ.float()).sum())
            mae_n += int(succ.sum())
        # Terminal-window frames (the last TERMINAL_WINDOW of each episode).
        term = (batch["t_index"].float() - batch["episode_len"].float()).abs() <= TERMINAL_WINDOW
        succ_term += v[term & succ].tolist()
        fail_term += v[term & ~succ].tolist()
    model.train()

    out = {"progress_mae": mae_sum / max(1, mae_n)}
    if succ_term:
        out["success_terminal_value"] = float(sum(succ_term) / len(succ_term))
    if fail_term:
        out["failure_terminal_value"] = float(sum(fail_term) / len(fail_term))
    if succ_term and fail_term:
        out["terminal_auroc"] = _auroc(succ_term, fail_term)
    return out


def _auroc(pos: list[float], neg: list[float]) -> float:
    """AUROC of positive (success-terminal) vs negative (failure-terminal) values
    via the Mann-Whitney U statistic (rank-based, no sklearn dependency)."""
    scores = [(s, 1) for s in pos] + [(s, 0) for s in neg]
    scores.sort(key=lambda x: x[0])
    rank_sum = 0.0
    i = 0
    n = len(scores)
    while i < n:  # average ranks within ties
        j = i
        while j < n and scores[j][0] == scores[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-based average rank of the tie block
        for k in range(i, j):
            if scores[k][1] == 1:
                rank_sum += avg_rank
        i = j
    n_pos, n_neg = len(pos), len(neg)
    u = rank_sum - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def _make_loader(episodes, collator, batch, workers, shuffle):
    ds = ProgressTransitionDataset(episodes)
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=shuffle,
        collate_fn=collator,
        persistent_workers=workers > 0,
    )


def main() -> None:
    args = build_argparser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    # Model first: its loaded SmolVLA policy drives the collator's preprocessing.
    model = SmolVLAProgressModel(ProgressModelConfig(
        smolvla_checkpoint=args.smolvla_checkpoint
    )).to(device)
    print(f"progress model params: {count_parameters(model)}")
    collator = SmolVLAProgressCollator(model.policy)

    episodes = build_progress_episodes(args.pi_root, args.nvidia_root)
    splits = grouped_split(episodes, args.val_fraction, args.test_fraction, args.seed)
    train_loader = _make_loader(
        splits["train"], collator, args.batch, args.workers, shuffle=True
    )
    val_loader = _make_loader(
        splits["val"], collator, args.batch, max(1, args.workers // 2), shuffle=False
    )

    optim = torch.optim.AdamW(
        model.param_groups(args.value_lr, args.expert_lr),
        weight_decay=args.weight_decay,
    )

    start_step = 0
    ema: EMATarget | None = None
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_trainable_state_dict(ckpt["trainable"])
        start_step = int(ckpt.get("step", 0))
        if start_step >= args.stage2_step:
            model.unfreeze_expert()
        if start_step >= args.stage3_step:
            ema = EMATarget(model)
        print(f"resumed from {args.resume} @ step {start_step}")

    (args.output / "config.json").write_text(
        json.dumps(
            {"progress_cfg": ProgressModelConfig(
                smolvla_checkpoint=args.smolvla_checkpoint).__dict__,
             "args": {k: str(v) for k, v in vars(args).items()}},
            indent=2,
        )
    )

    amp_device = "cuda" if device == "cuda" else "cpu"
    best_mae = float("inf")
    step = start_step
    t0 = time.time()
    running = {"loss": 0.0, "progress": 0.0, "td": 0.0, "n": 0}
    model.train()

    def stage_of(s: int) -> int:
        return 3 if s >= args.stage3_step else 2 if s >= args.stage2_step else 1

    data_iter = iter(train_loader)
    while step < args.steps:
        # Stage transitions (idempotent each step; cheap to re-check).
        if step == args.stage2_step and not model.expert_trainable:
            model.unfreeze_expert()
            print(f"[stage 2] unfroze action expert @ step {step}")
        if step == args.stage3_step and ema is None:
            ema = EMATarget(model)
            print(f"[stage 3] initialized EMA target @ step {step}")

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        batch = _batch_to_device(batch, device)

        # Per-group cosine-with-warmup LR (two bases, same schedule shape).
        for g, base in zip(optim.param_groups, (args.value_lr, args.expert_lr)):
            g["lr"] = lr_at(step, base, args.warmup_steps, args.steps)

        use_td = step >= args.stage3_step
        with torch.autocast(device_type=amp_device, dtype=torch.bfloat16):
            v = model(batch["images"], batch["img_masks"],
                      batch["lang_tokens"], batch["lang_masks"])
            l_prog, m_prog = progress_regression_loss(
                v.float(), batch["t_index"], batch["episode_len"],
                batch["success_mask"],
            )
            loss = l_prog
            m_td = {"td_loss": 0.0}
            if use_td:
                next_v = ema.value(
                    batch["next_images"], batch["next_img_masks"],
                    batch["lang_tokens"], batch["lang_masks"],
                ).float()
                l_td, m_td = td_loss(
                    v.float(), next_v, batch["t_index"], batch["episode_len"],
                    batch["success_mask"], args.gamma, args.terminal_window,
                )
                loss = l_prog + l_td

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), args.grad_clip
        )
        optim.step()
        if use_td:
            ema.update(model, args.tau)

        bsz = v.shape[0]
        running["loss"] += float(loss.detach()) * bsz
        running["progress"] += m_prog["progress_loss"] * bsz
        running["td"] += m_td["td_loss"] * bsz
        running["n"] += bsz

        if step % args.log_interval == 0 and running["n"] > 0:
            ips = (step - start_step + 1) * args.batch / max(time.time() - t0, 1e-6)
            print(
                f"step {step:>7d} | stage {stage_of(step)} | "
                f"loss {running['loss'] / running['n']:.5f} | "
                f"prog {running['progress'] / running['n']:.5f} | "
                f"td {running['td'] / running['n']:.5f} | "
                f"lr_v {optim.param_groups[0]['lr']:.2e} | {ips:.0f} samp/s"
            )
            running = {"loss": 0.0, "progress": 0.0, "td": 0.0, "n": 0}

        if step > 0 and step % args.val_interval == 0:
            val = run_validation(model, val_loader, device, args.val_batches)
            print(f"  [val @ {step}] {json.dumps(val)}")
            if val["progress_mae"] < best_mae:
                best_mae = val["progress_mae"]
                model.save_checkpoint(str(args.output / "best.pt"), step=step, extra={"val": val})
                print(f"  saved best (progress_mae={best_mae:.5f})")

        if step > 0 and step % args.save_interval == 0:
            model.save_checkpoint(str(args.output / "latest.pt"), step=step)

        step += 1

    model.save_checkpoint(str(args.output / "final.pt"), step=step)
    print(f"done. best progress MAE: {best_mae:.5f}")

    if args.hf_repo_id:
        upload_to_huggingface(
            output_dir=args.output,
            repo_id=args.hf_repo_id,
            token=args.hf_token,
            private=args.hf_private,
        )
        print(f"uploaded progress checkpoint folder to https://huggingface.co/{args.hf_repo_id}")


if __name__ == "__main__":
    main()
