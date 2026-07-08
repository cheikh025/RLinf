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

"""Record short random-action RoboCasa smoke-test videos.

This script verifies that RoboCasa task names are registered and renderable without
loading an RLinf policy checkpoint.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


DEFAULT_TASKS = [
    "CloseDoubleDoor",
    "CloseDrawer",
    "CloseSingleDoor",
    "CoffeePressButton",
    "CoffeeServeMug",
    "CoffeeSetupMug",
    "OpenDoubleDoor",
    "OpenDrawer",
    "OpenSingleDoor",
    "PnPCabToCounter",
    "PnPCounterToCab",
    "PnPCounterToMicrowave",
    "PnPCounterToSink",
    "PnPCounterToStove",
    "PnPMicrowaveToCounter",
    "PnPSinkToCounter",
    "PnPStoveToCounter",
    "TurnOffMicrowave",
    "TurnOffSinkFaucet",
    "TurnOffStove",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
    "TurnOnStove",
    "TurnSinkSpout",
]

DEFAULT_CAMERAS = [
    "robot0_agentview_left",
    "robot0_eye_in_hand",
    "robot0_agentview_right",
]


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def _as_uint8_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        if array.max(initial=0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return array


def _extract_camera_frame(obs: dict, camera_names: list[str]) -> np.ndarray:
    frames = []
    for camera_name in camera_names:
        obs_key = f"{camera_name}_image"
        if obs_key not in obs:
            continue
        # Robosuite camera observations are vertically flipped relative to display.
        frames.append(_as_uint8_image(obs[obs_key])[::-1])
    if not frames:
        raise KeyError(
            f"No camera observations found. Expected keys like "
            f"{[f'{name}_image' for name in camera_names]}"
        )
    return np.concatenate(frames, axis=1)


def _annotate(frame: np.ndarray, title: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    text_h = 20
    canvas = Image.new("RGB", (image.width, image.height + text_h), (0, 0, 0))
    canvas.paste(image, (0, text_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), title, fill=(255, 255, 255))
    return np.asarray(canvas)


def _sample_action(env, rng: np.random.Generator, action_scale: float) -> np.ndarray:
    low, high = env.action_spec
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    low = np.nan_to_num(low, nan=-1.0, neginf=-1.0, posinf=1.0)
    high = np.nan_to_num(high, nan=1.0, neginf=-1.0, posinf=1.0)
    action = rng.uniform(low, high).astype(np.float32)
    return np.clip(action * action_scale, low, high)


def _make_env(args: argparse.Namespace, task: str):
    import robocasa  # noqa: F401  Import registers RoboCasa envs and robots.
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    controller_config = load_composite_controller_config(
        controller=None,
        robot=args.robot,
    )
    return robosuite.make(
        env_name=task,
        robots=args.robot,
        controller_configs=controller_config,
        camera_names=args.cameras,
        camera_widths=args.width,
        camera_heights=args.height,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=args.seed,
        translucent_robot=False,
        render_camera=args.render_camera,
    )


def record_task(args: argparse.Namespace, task: str, out_dir: Path) -> tuple[bool, str]:
    rng = np.random.default_rng(args.seed)
    env = None
    try:
        env = _make_env(args, task)
        obs = env.reset()
        meta = env.get_ep_meta() if hasattr(env, "get_ep_meta") else {}
        lang = meta.get("lang", "")

        frames = []
        title = f"{task}: {lang}" if lang else task
        reset_frame = _annotate(_extract_camera_frame(obs, args.cameras), title)
        frames.append(reset_frame)

        task_dir = out_dir / _slug(task)
        task_dir.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(task_dir / "reset.png", reset_frame)

        for step_idx in range(args.steps):
            action = _sample_action(env, rng, args.action_scale)
            obs, reward, done, info = env.step(action)
            frame_title = f"{title} | step={step_idx + 1} reward={float(reward):.3f}"
            frames.append(_annotate(_extract_camera_frame(obs, args.cameras), frame_title))

        imageio.mimsave(task_dir / "random_actions.mp4", frames, fps=args.fps)
        return True, lang
    except Exception as exc:  # noqa: BLE001 - smoke-test should report all failures.
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record short random-action videos for RoboCasa tasks."
    )
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--out-dir", default="robocasa_task_smoke_outputs")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--robot", default="PandaOmron")
    parser.add_argument("--cameras", nargs="+", default=DEFAULT_CAMERAS)
    parser.add_argument("--render-camera", default="robot0_agentview_center")
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument(
        "--action-scale",
        type=float,
        default=0.3,
        help="Scale random actions after sampling from the environment action bounds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing outputs to {out_dir}")
    failures = []
    for task in args.tasks:
        ok, detail = record_task(args, task, out_dir)
        if ok:
            print(f"OK   {task}: {detail}")
        else:
            print(f"FAIL {task}: {detail}")
            failures.append(task)

    if failures:
        raise SystemExit(f"Failed tasks: {', '.join(failures)}")


if __name__ == "__main__":
    main()
