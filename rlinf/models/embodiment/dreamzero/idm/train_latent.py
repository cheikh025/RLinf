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

"""Latent-space IDM training entrypoint (single GPU).

Trains :class:`...idm.latent_model.LatentActionIDM` on WAN-VAE latents of real
clips -- the same latent space the DiT produces, so there is no decoder leg in
the train/inference gap (design record: ``dreamzero_latent_idm_design_README.md``).

Two data paths (one flag):

- ``--cache-dir DIR`` (recommended): train from the fp32 latent cache under
  ``DIR/train`` and ``DIR/val``. If those caches do not exist they are built
  first with :func:`...idm.latent_data.precompute_latent_cache` (needs
  ``--vae-path`` and ``--data``); once built, the VAE is never touched again.
- no ``--cache-dir``: encode on the fly with
  :class:`...idm.latent_data.LatentPrefetcher` (needs ``--vae-path``); 0 disk,
  re-encodes every epoch behind the IDM step.

Example (cached, builds the cache on first run):

    python -m rlinf.models.embodiment.dreamzero.idm.train_latent \
        --data physical-intelligence/libero \
        --vae-path /path/Wan2.2_VAE.pth \
        --cache-dir ./cache/idm_latent \
        --output ./checkpoints/idm_latent --steps 100000 --batch 64

Recipe matches the pixel IDM: AdamW + cosine warmup, bf16 autocast, grad clip
1.0, loss = SmoothL1(beta=0.1, standardized arm) + lambda_grip * weighted BCE.
Arm standardization and the gripper ``pos_weight`` come from the train split
(the cache stores them in ``meta.json``) and are baked into the checkpoint.

Validation reports the Gate-1 dashboard: per-dim env-space RMSE, binary gripper
accuracy, and the jerk ratio (predicted / ground-truth RMS second difference) --
the regression-to-the-mean alarm.
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rlinf.models.embodiment.dreamzero.idm.latent_data import (
    CachedLatentDataset,
    IDMLatentDataset,
    LatentEncoder,
    LatentPrefetcher,
    collate_idm,
    collate_latent,
    precompute_latent_cache,
)
from rlinf.models.embodiment.dreamzero.idm.latent_model import (
    LatentActionIDM,
    LatentIDMConfig,
    compute_loss,
    count_parameters,
)

# Shared, model-agnostic helpers (no copy-paste).
from rlinf.models.embodiment.dreamzero.idm.train import (
    _rms_jerk,
    lr_at,
    upload_to_huggingface,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        default=None,
        help=(
            "LeRobot dataset path/repo id (same as SFT data.train_data_paths). "
            "Required for on-the-fly mode and for building a missing cache; not "
            "needed if --cache-dir already contains built caches."
        ),
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory holding the fp32 latent cache (train/ and val/ subdirs). "
            "Built on first run if missing (needs --vae-path and --data); "
            "thereafter the VAE is never loaded."
        ),
    )
    p.add_argument(
        "--vae-path",
        default=None,
        help=(
            "Path to Wan2.2_VAE.pth (z_dim=48). Required to encode latents "
            "(on-the-fly mode, or building the cache)."
        ),
    )
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=2_000)
    p.add_argument("--lambda-grip", type=float, default=0.05)
    p.add_argument("--arm-beta", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--val-interval", type=int, default=2_000)
    p.add_argument("--val-batches", type=int, default=50)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--save-interval", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--cache-batch", type=int, default=32, help="Encode batch size.")
    p.add_argument("--video-backend", default="pyav")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--resume", type=Path, default=None, help="Checkpoint to resume from."
    )
    p.add_argument(
        "--hf-repo-id",
        default=None,
        help="Optional Hugging Face model repo id to upload the output folder.",
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


def _device_batches(loader: DataLoader, device: str):
    """Yield ``(latent, actions)`` already on device from a cached loader."""
    for batch in loader:
        yield (
            batch["latent"].to(device, non_blocking=True),
            batch["actions"].to(device, non_blocking=True),
        )


@torch.no_grad()
def run_validation(
    model: LatentActionIDM,
    val_iter_factory,
    device: str,
    max_batches: int,
    lambda_grip: float,
    arm_beta: float,
    pos_weight: torch.Tensor,
) -> dict:
    """Gate-1 dashboard on the held-out split (env-space metrics).

    ``val_iter_factory`` returns a fresh iterator of ``(latent, target)`` on
    device (cached batches or a one-shot prefetcher), so validation can run
    repeatedly during training.
    """
    model.eval()
    n = 0
    loss_sum = 0.0
    per_dim_sq = torch.zeros(model.cfg.action_dim - 1, dtype=torch.float64)
    grip_correct = 0
    grip_total = 0
    jerk_pred_sum = 0.0
    jerk_gt_sum = 0.0
    for i, (latent, target) in enumerate(val_iter_factory()):
        if i >= max_batches:
            break
        out = model(latent)
        loss, _ = compute_loss(
            out,
            target,
            model,
            lambda_grip=lambda_grip,
            arm_beta=arm_beta,
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


def _build_cache_if_missing(args, device: str) -> tuple[Path, Path]:
    """Ensure ``cache-dir/{train,val}`` exist; build them with the VAE if not."""
    train_dir = args.cache_dir / "train"
    val_dir = args.cache_dir / "val"
    if (train_dir / "meta.json").exists() and (val_dir / "meta.json").exists():
        print(f"using existing latent cache at {args.cache_dir}")
        return train_dir, val_dir

    if not args.vae_path:
        raise ValueError("building the latent cache requires --vae-path.")
    if not args.data:
        raise ValueError("building the latent cache requires --data.")

    encoder = LatentEncoder(args.vae_path, device=device)
    for split, cdir in (("train", train_dir), ("val", val_dir)):
        ds = IDMLatentDataset(
            data_path=args.data,
            split=split,
            skip_invalid=True,
            val_fraction=args.val_fraction,
            split_seed=args.seed,
            video_backend=args.video_backend,
        )
        print(f"building latent cache: split={split} anchors={len(ds)} -> {cdir}")
        meta = precompute_latent_cache(
            ds,
            encoder,
            str(cdir),
            batch_size=args.cache_batch,
            num_workers=args.workers,
            device=device,
        )
        print(f"  cached {meta['count']} latents {meta['latent_shape']} (fp32)")
    del encoder  # free the VAE before training
    if device == "cuda":
        torch.cuda.empty_cache()
    return train_dir, val_dir


def main() -> None:
    args = build_argparser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    encoder = None
    if args.cache_dir is not None:
        train_dir, val_dir = _build_cache_if_missing(args, device)
        train_cache = CachedLatentDataset(str(train_dir))
        val_cache = CachedLatentDataset(str(val_dir))
        stats = train_cache.action_stats
        latent_shape = train_cache.latent_shape
        n_train, n_val = len(train_cache), len(val_cache)

        train_loader = DataLoader(
            train_cache,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_latent,
            persistent_workers=args.workers > 0,
        )
        val_loader = DataLoader(
            val_cache,
            batch_size=args.batch,
            shuffle=False,
            num_workers=max(1, args.workers // 2),
            pin_memory=True,
            collate_fn=collate_latent,
            persistent_workers=args.workers > 0,
        )
        train_iter_factory = lambda: _device_batches(train_loader, device)  # noqa: E731
        val_iter_factory = lambda: _device_batches(val_loader, device)  # noqa: E731
    else:
        if not args.vae_path:
            raise ValueError("on-the-fly mode requires --vae-path (or use --cache-dir).")
        if not args.data:
            raise ValueError("--data is required.")
        common = {
            "data_path": args.data,
            "val_fraction": args.val_fraction,
            "split_seed": args.seed,
            "video_backend": args.video_backend,
        }
        train_ds = IDMLatentDataset(split="train", **common)
        val_ds = IDMLatentDataset(split="val", **common)
        stats = train_ds.compute_action_stats()
        latent_shape = (48, 3, 10, 20)  # WanVideoVAE38 z_dim=48 on the 160x320 canvas
        n_train, n_val = len(train_ds), len(val_ds)

        encoder = LatentEncoder(args.vae_path, device=device)
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
        train_iter_factory = lambda: iter(  # noqa: E731
            LatentPrefetcher(train_loader, encoder, device)
        )
        val_iter_factory = lambda: iter(  # noqa: E731
            LatentPrefetcher(val_loader, encoder, device)
        )

    print(f"train samples: {n_train}; val samples: {n_val}")
    print(f"action stats: {json.dumps(stats)}")
    pos_weight = torch.tensor(stats["gripper_pos_weight"], device=device)

    c, t, h, w = latent_shape
    idm_cfg = LatentIDMConfig(
        latent_channels=c, latent_t=t, latent_h=h, latent_w=w
    )
    model = LatentActionIDM(idm_cfg).to(device)
    model.set_action_stats(
        torch.tensor(stats["arm_mean"]), torch.tensor(stats["arm_std"])
    )
    print(f"Latent IDM params: {count_parameters(model) / 1e6:.1f}M cfg={idm_cfg}")

    start_step = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_step = int(ckpt.get("step", 0))
        print(f"resumed from {args.resume} @ step {start_step}")

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
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

    amp_device = "cuda" if device == "cuda" else "cpu"
    best_val = float("inf")
    step = start_step
    data_iter = train_iter_factory()
    t0 = time.time()
    running_loss = 0.0
    running_n = 0
    model.train()

    while step < args.steps:
        try:
            latent, target = next(data_iter)
        except StopIteration:
            data_iter = train_iter_factory()
            latent, target = next(data_iter)

        lr = lr_at(step, args.lr, args.warmup_steps, args.steps)
        for g in optim.param_groups:
            g["lr"] = lr

        with torch.autocast(device_type=amp_device, dtype=torch.bfloat16):
            out = model(latent)
            loss, metrics = compute_loss(
                out,
                target,
                model,
                lambda_grip=args.lambda_grip,
                arm_beta=args.arm_beta,
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
                val_iter_factory,
                device,
                args.val_batches,
                args.lambda_grip,
                args.arm_beta,
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
        print(f"uploading Latent IDM output folder to Hugging Face: {args.hf_repo_id}")
        upload_to_huggingface(
            output_dir=args.output,
            repo_id=args.hf_repo_id,
            token=args.hf_token,
            private=args.hf_private,
        )
        print(
            "uploaded Latent IDM checkpoint folder to "
            f"https://huggingface.co/{args.hf_repo_id}"
        )


if __name__ == "__main__":
    main()
