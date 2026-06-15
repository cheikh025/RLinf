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

"""Empirically verify the Wan2.2 VAE used by DreamZero is deterministic.

Code reading (rlinf patch, dreamzero0/dreamzero, Wan-Video/Wan2.2) says
``encode()`` returns the posterior mean ``mu`` and never samples; this script
proves it at runtime on the exact class RLinf uses
(``rlinf.models.embodiment.dreamzero.patch.wan_video_vae.WanVideoVAE38``,
z_dim=48, weights ``Wan2.2_VAE.pth``, CUDA bf16 -- mirroring the action head).

Three independent checks, per tiling mode:

1. **Seed independence** -- encode the same input under different global
   seeds. Bit-identical outputs => no random sampling in the forward path.
2. **RNG consumption** -- snapshot CPU+CUDA RNG state around encode/decode.
   Unchanged state => ``randn``/``rand`` is never even called.
3. **Repeatability** -- back-to-back identical calls catch nondeterministic
   kernels (atomics, cuDNN autotune), which is a separate failure mode from
   sampling.

Also reports the encode(decode(z)) roundtrip drift, which bounds the error of
using one offline VAE roundtrip as the IDM training-data transform.

Run on the GPU box inside the dreamzero venv (the policy's environment):

    python dreamzero_verify_vae_determinism.py \
        --vae-path /path/to/Wan2.2_VAE.pth

Expected verdict line: ``VAE_DETERMINISTIC`` (exit code 0).
"""

import argparse

import torch


def _report(name: str, a: torch.Tensor, b: torch.Tensor) -> bool:
    identical = torch.equal(a, b)
    max_diff = (a.float() - b.float()).abs().max().item()
    print(f"  {name}: bitwise_identical={identical} max_abs_diff={max_diff:.3e}")
    return identical


def _rng_states() -> tuple:
    return (torch.get_rng_state().clone(), torch.cuda.get_rng_state_all())


def _rng_states_equal(s1: tuple, s2: tuple) -> bool:
    cpu_ok = torch.equal(s1[0], s2[0])
    cuda_ok = len(s1[1]) == len(s2[1]) and all(
        torch.equal(a, b) for a, b in zip(s1[1], s2[1])
    )
    return cpu_ok and cuda_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vae-path", required=True, help="Path to Wan2.2_VAE.pth (z_dim=48)."
    )
    parser.add_argument(
        "--frames", type=int, default=9, help="Input frames (9 mirrors production)."
    )
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp32"],
        help="bf16 mirrors the action head's VAE dtype.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required (WanVideoVAE38 hardcodes cuda buffers).")
        return 2

    # Prefer the exact class RLinf runs (patched); fall back to groot's own
    # WanVideoVAE38 so the script also works in a bare torch+groot env without
    # rlinf/ray installed. The patch only adds batching/tiling around the same
    # groot ``VideoVAE38_`` forward path, so the determinism verdict is
    # identical either way.
    try:
        from rlinf.models.embodiment.dreamzero.patch.wan_video_vae import (
            WanVideoVAE38,
        )

        print("using rlinf patched WanVideoVAE38")
    except ImportError:
        from groot.vla.model.dreamzero.modules.wan_video_vae import WanVideoVAE38

        print("rlinf not importable; using groot WanVideoVAE38")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    vae = WanVideoVAE38(z_dim=48)
    state_dict = torch.load(args.vae_path, map_location="cpu")
    vae.model.load_state_dict(state_dict)  # same load as WANPolicyHead
    vae.to(device="cuda", dtype=dtype).eval()

    # Fixed input in [-1, 1], production shape [B, C, T, H, W]; the CPU
    # generator pins the input itself regardless of global seeds.
    gen = torch.Generator().manual_seed(7)
    x = torch.rand((1, 3, args.frames, args.height, args.width), generator=gen)
    x = (x * 2 - 1).to("cuda", dtype)

    all_ok = True
    for tiled in (False, True):
        print(f"[tiled={tiled}]")

        def enc(t: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return vae.encode(t, tiled=tiled)

        def dec(z: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return vae.decode(z, tiled=tiled)

        # 1. Seed independence: different global seeds, same input.
        torch.manual_seed(0)
        z1 = enc(x)
        torch.manual_seed(999_999)
        z2 = enc(x)
        all_ok &= _report("encode under seed 0 vs seed 999999", z1, z2)

        # 2. RNG consumption: state must not advance across encode+decode.
        torch.manual_seed(123)
        before = _rng_states()
        _ = enc(x)
        d1 = dec(z1)
        rng_untouched = _rng_states_equal(before, _rng_states())
        print(f"  rng_state_untouched_by_encode_decode={rng_untouched}")
        all_ok &= rng_untouched

        # 3. Repeatability (kernel-level): back-to-back identical calls.
        all_ok &= _report("encode repeat same process", z2, enc(x))
        torch.manual_seed(31_337)
        d2 = dec(z1)
        all_ok &= _report("decode under seed 123 vs seed 31337", d1, d2)
        all_ok &= _report("decode repeat same process", d2, dec(z1))

        # Informational: roundtrip drift z -> decode -> encode -> z'.
        z_rt = enc(dec(z1))
        drift = (z_rt.float() - z1.float()).abs().max().item()
        rel = drift / z1.float().abs().max().clamp_min(1e-8).item()
        print(f"  roundtrip latent drift: max_abs={drift:.3e} rel={rel:.3e}")

    print("VAE_DETERMINISTIC" if all_ok else "VAE_NONDETERMINISTIC")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
