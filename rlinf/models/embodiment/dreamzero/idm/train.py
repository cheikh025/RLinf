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

"""IDM training entrypoint (DreamZero PRM Milestone 3, single GPU).

Run on the GPU box inside the dreamzero venv:

    python -m rlinf.models.embodiment.dreamzero.idm.train \
        --data physical-intelligence/libero \
        --output ./checkpoints/idm_libero \
        --steps 100000 --batch 32

Recipe (design record: ``dreamzero_prm_milestone3_idm_README.md``):
AdamW + cosine with warmup, bf16 autocast, grad clip 1.0;
loss = MSE(standardized arm) + lambda_grip * weighted BCE. The IDM predicts
the full 24-action chunk covering the 9-frame window's 24-step span; the
consistency scorer uses only the first 16 (the WAM chunk). Arm
standardization stats and the gripper ``pos_weight`` are computed once
from the train split and baked into the model buffers / checkpoint.

Validation reports the Gate-1 dashboard: per-dim env-space RMSE, binary
gripper accuracy, and the jerk ratio (predicted / ground-truth RMS of the
second action difference) -- the regression-to-the-mean alarm: a ratio well
below 1 means the IDM over-smooths and ``--lambda-grip`` / the MSE arm loss
need revisiting.
"""

import argparse
import dataclasses
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rlinf.models.embodiment.dreamzero.idm.data import IDMChunkDataset, collate_idm
from rlinf.models.embodiment.dreamzero.idm.model import (
    IDM,
    IDMConfig,
    compute_loss,
    count_parameters,
)
from rlinf.models.embodiment.dreamzero.idm.vae_roundtrip import (
    RoundtripPrefetcher,
    VaeRoundtrip,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        required=True,
        help="LeRobot dataset path/repo id (same as SFT data.train_data_paths).",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=2_000)
    p.add_argument("--lambda-grip", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--val-interval", type=int, default=2_000)
    p.add_argument("--val-batches", type=int, default=50)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--save-interval", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument(
        "--late-fusion",
        action="store_true",
        help="Ablation: single-frame late fusion instead of pairwise early fusion.",
    )
    p.add_argument(
        "--train-aug",
        action="store_true",
        help="SFT-style random crop + color jitter on training clips.",
    )
    p.add_argument(
        "--scratch-backbone",
        action="store_true",
        help="Ablation: random-init ResNet instead of ImageNet pretrained.",
    )
    p.add_argument("--video-backend", default="pyav")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--vae-roundtrip",
        action="store_true",
        help=(
            "On-the-fly WAN-VAE decode(encode()) of training frames so the IDM "
            "trains on the dream-domain artifact distribution it sees at "
            "inference (domain-gap mitigation). Overlapped on a side CUDA "
            "stream; no disk cost."
        ),
    )
    p.add_argument(
        "--vae-path",
        default=None,
        help="Path to Wan2.2_VAE.pth (z_dim=48); required with --vae-roundtrip.",
    )
    p.add_argument(
        "--vae-roundtrip-prob",
        type=float,
        default=1.0,
        help=(
            "Fraction of batches to roundtrip (1.0 = every batch). <1 keeps "
            "some clean frames for real-frame accuracy; this is a modeling "
            "knob, not a memory one."
        ),
    )
    p.add_argument(
        "--resume", type=Path, default=None, help="Checkpoint to resume from."
    )
    p.add_argument(
        "--hf-repo-id",
        default=None,
        help=(
            "Optional Hugging Face model repo id to upload the output folder "
            "after training, e.g. cheikh025/dreamzero-idm-libero."
        ),
    )
    p.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token. Defaults to HF_TOKEN from the environment.",
    )
    p.add_argument(
        "--hf-private",
        action="store_true",
        help="Create the Hugging Face repo as private if it does not already exist.",
    )
    return p


def lr_at(step: int, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _rms_jerk(arm_actions: torch.Tensor) -> torch.Tensor:
    """RMS second difference over the chunk; delta actions are velocity-level,
    so diff^2 is the jerk analogue (matches the M2 executability convention)."""
    jerk = arm_actions[:, 2:] - 2 * arm_actions[:, 1:-1] + arm_actions[:, :-2]
    return (jerk**2).mean(dim=(1, 2)).sqrt()


def upload_to_huggingface(
    output_dir: Path,
    repo_id: str,
    token: str | None,
    private: bool,
) -> None:
    """Upload the IDM output directory to a Hugging Face model repository."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face upload requested, but huggingface_hub is not installed. "
            "Install it in the training VM with `pip install -U huggingface_hub`."
        ) from exc

    auth_token = token or os.getenv("HF_TOKEN")
    if not auth_token:
        raise RuntimeError(
            "Hugging Face upload requested, but no token was provided. "
            "Pass --hf-token or set HF_TOKEN in the training VM."
        )

    api = HfApi(token=auth_token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(output_dir),
        path_in_repo=".",
        commit_message="Upload DreamZero IDM checkpoint",
    )


@torch.no_grad()
def run_validation(
    model: IDM,
    loader: DataLoader,
    device: str,
    max_batches: int,
    lambda_grip: float,
    pos_weight: torch.Tensor,
) -> dict:
    """Gate-1 dashboard on the held-out split (env-space metrics)."""
    model.eval()
    n = 0
    loss_sum = 0.0
    per_dim_sq = torch.zeros(model.cfg.action_dim - 1, dtype=torch.float64)
    grip_correct = 0
    grip_total = 0
    jerk_pred_sum = 0.0
    jerk_gt_sum = 0.0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        video = batch["video"].to(device, non_blocking=True)
        target = batch["actions"].to(device, non_blocking=True)
        out = model(video)
        loss, _ = compute_loss(
            out,
            target,
            model,
            lambda_grip=lambda_grip,
            gripper_pos_weight=pos_weight,
        )
        bsz = target.shape[0]
        loss_sum += loss.item() * bsz
        n += bsz

        arm_env = out["arm"].float() * model.arm_std + model.arm_mean
        arm_gt = target[..., :-1].float()
        per_dim_sq += ((arm_env - arm_gt) ** 2).mean(dim=(0, 1)).double().cpu() * bsz

        pred_bin = out["gripper_logit"] > 0
        gt_bin = target[..., -1] > 0
        grip_correct += int((pred_bin == gt_bin).sum().item())
        grip_total += int(pred_bin.numel())

        jerk_pred_sum += _rms_jerk(arm_env).sum().item()
        jerk_gt_sum += _rms_jerk(arm_gt).sum().item()
    model.train()
    if n == 0:
        return {"val_loss": float("nan")}
    return {
        "val_loss": loss_sum / n,
        "per_dim_rmse": (per_dim_sq / n).sqrt().tolist(),
        "gripper_accuracy": grip_correct / max(1, grip_total),
        "jerk_ratio_pred_over_gt": (jerk_pred_sum / n) / max(1e-9, jerk_gt_sum / n),
    }


def main() -> None:
    args = build_argparser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    common = {
        "data_path": args.data,
        "val_fraction": args.val_fraction,
        "split_seed": args.seed,
        "video_backend": args.video_backend,
    }
    train_ds = IDMChunkDataset(split="train", train_aug=args.train_aug, **common)
    val_ds = IDMChunkDataset(split="val", **common)
    print(f"train anchors: {len(train_ds)}; val anchors: {len(val_ds)}")

    stats = train_ds.compute_action_stats()
    print(f"action stats: {json.dumps(stats)}")
    pos_weight = torch.tensor(stats["gripper_pos_weight"], device=device)

    idm_cfg = IDMConfig(
        pair_fusion=not args.late_fusion,
        pretrained_backbone=not args.scratch_backbone,
    )
    model = IDM(idm_cfg).to(device)
    model.set_action_stats(
        torch.tensor(stats["arm_mean"]), torch.tensor(stats["arm_std"])
    )
    print(f"IDM params: {count_parameters(model) / 1e6:.1f}M cfg={idm_cfg}")

    start_step = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_step = int(ckpt.get("step", 0))
        print(f"resumed from {args.resume} @ step {start_step}")

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_idm,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=max(1, args.workers // 2),
        pin_memory=True,
        collate_fn=collate_idm,
        persistent_workers=args.workers > 0,
    )

    (args.output / "config.json").write_text(
        json.dumps(
            {
                "idm_cfg": dataclasses.asdict(idm_cfg),
                "stats": stats,
                "args": {k: str(v) for k, v in vars(args).items()},
            },
            indent=2,
        )
    )

    def save_ckpt(path: Path, step: int, val: dict | None) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "idm_cfg": dataclasses.asdict(idm_cfg),
                "stats": stats,
                "step": step,
                "val": val,
            },
            path,
        )

    roundtrip = None
    if args.vae_roundtrip:
        if not args.vae_path:
            raise ValueError("--vae-roundtrip requires --vae-path (Wan2.2_VAE.pth).")
        roundtrip = VaeRoundtrip(args.vae_path, device=device)
        print(
            f"VAE roundtrip enabled: prob={args.vae_roundtrip_prob} path={args.vae_path}"
        )
    prefetcher = RoundtripPrefetcher(
        train_loader, roundtrip, device, prob=args.vae_roundtrip_prob
    )

    best_val = float("inf")
    step = start_step
    data_iter = iter(prefetcher)
    t0 = time.time()
    running_loss = 0.0
    running_n = 0
    model.train()

    while step < args.steps:
        try:
            video, target = next(data_iter)
        except StopIteration:
            data_iter = iter(prefetcher)
            video, target = next(data_iter)

        lr = lr_at(step, args.lr, args.warmup_steps, args.steps)
        for g in optim.param_groups:
            g["lr"] = lr

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(video)
            loss, metrics = compute_loss(
                out,
                target,
                model,
                lambda_grip=args.lambda_grip,
                gripper_pos_weight=pos_weight,
            )

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        running_loss += metrics["loss"] * target.shape[0]
        running_n += target.shape[0]

        if step % args.log_interval == 0 and running_n > 0:
            ips = (step - start_step + 1) * args.batch / max(time.time() - t0, 1e-6)
            print(
                f"step {step:>7d} | loss {running_loss / running_n:.5f} "
                f"| arm {metrics['arm_loss']:.5f} | grip {metrics['grip_loss']:.4f} "
                f"| acc {metrics['grip_acc']:.3f} | lr {lr:.2e} | {ips:.0f} samp/s"
            )
            running_loss = 0.0
            running_n = 0

        if step > 0 and step % args.val_interval == 0:
            val = run_validation(
                model,
                val_loader,
                device,
                args.val_batches,
                args.lambda_grip,
                pos_weight,
            )
            print(f"  [val @ {step}] {json.dumps(val)}")
            if val["val_loss"] < best_val:
                best_val = val["val_loss"]
                save_ckpt(args.output / "best.pt", step, val)
                print(f"  saved best (val_loss={best_val:.5f})")

        if step > 0 and step % args.save_interval == 0:
            save_ckpt(args.output / "latest.pt", step, None)

        step += 1

    save_ckpt(args.output / "final.pt", step, None)
    print(f"done. best val loss: {best_val:.5f}")

    if args.hf_repo_id:
        print(f"uploading IDM output folder to Hugging Face: {args.hf_repo_id}")
        upload_to_huggingface(
            output_dir=args.output,
            repo_id=args.hf_repo_id,
            token=args.hf_token,
            private=args.hf_private,
        )
        print(
            "uploaded IDM checkpoint folder to "
            f"https://huggingface.co/{args.hf_repo_id}"
        )


if __name__ == "__main__":
    main()
