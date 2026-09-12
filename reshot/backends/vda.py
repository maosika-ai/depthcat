"""Video Depth Anything backend (default).

Why this model: it is the only open video-depth model that ships a temporal module
*and* an Apache-2.0 checkpoint (Small). Per-frame models (Depth Anything V2/3) flicker
on video unless you bolt on your own smoothing; diffusion models (DepthCrafter) are
~100× slower. VDA-Small at 28M params runs on an 8 GB consumer GPU or an Apple M-series.

Precision policy (measured 2026-09-11 on M2 Max, 32 frames @ 736×1280):
    mps  fp32 →  7.0 s        mps fp16 → did not finish in 3 min (autocast pathological)
    cpu  fp32 → 59 s
    Full 294-frame clip end-to-end on mps fp32: 147 s (~500 ms/frame incl. alignment).
So fp16 is enabled on CUDA only. Do not "optimise" this by turning fp16 on for MPS.

Licence policy: `small` is Apache-2.0. `base`/`large` are CC-BY-NC-4.0 — loading them
prints a warning and sets ``info.commercial_ok = False``; the CLI surfaces this.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import torch

from .base import BackendInfo

log = logging.getLogger(__name__)

_VARIANTS = {
    "small": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "base": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "large": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}
_LICENSES = {"small": "Apache-2.0", "base": "CC-BY-NC-4.0", "large": "CC-BY-NC-4.0"}
_HF_REPOS = {
    "small": "depth-anything/Video-Depth-Anything-Small",
    "base": "depth-anything/Video-Depth-Anything-Base",
    "large": "depth-anything/Video-Depth-Anything-Large",
}


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _weights_path(variant: str, checkpoint: str | None) -> str:
    if checkpoint:
        return checkpoint
    # huggingface_hub honours HF_ENDPOINT, so users in China can point it at a mirror
    # (e.g. https://hf-mirror.com) without us special-casing anything.
    from huggingface_hub import hf_hub_download

    filename = f"video_depth_anything_{_VARIANTS[variant]['encoder']}.pth"
    return hf_hub_download(_HF_REPOS[variant], filename)


class VDABackend:
    def __init__(
        self,
        model: str = "small",
        device: str = "auto",
        checkpoint: str | None = None,
    ) -> None:
        variant = model
        if variant not in _VARIANTS:
            from ..errors import BackendError

            raise BackendError(f"unknown model variant {variant!r}; choose from {sorted(_VARIANTS)}")
        from ..third_party.video_depth_anything.video_depth import VideoDepthAnything

        self.variant = variant
        self.device = pick_device(device)
        # fp16 only where it is known to be fast; see module docstring for the measurement.
        self.fp32 = self.device != "cuda"
        self.info = BackendInfo(
            name="video-depth-anything",
            variant=variant,
            license=_LICENSES[variant],
            commercial_ok=_LICENSES[variant].startswith("Apache"),
        )
        if not self.info.commercial_ok:
            warnings.warn(
                f"Video Depth Anything '{variant}' weights are {_LICENSES[variant]}: "
                "non-commercial use only. Use 'small' for commercial projects.",
                stacklevel=2,
            )

        try:
            path = _weights_path(variant, checkpoint)
        except Exception as exc:  # hub / network / offline-cache errors all land here
            from ..errors import BackendError

            raise BackendError(
                f"could not obtain weights for '{variant}': {exc}. "
                "Set HF_ENDPOINT to a mirror, or pass --checkpoint with a local .pth."
            ) from exc
        log.info("loading %s from %s on %s (fp32=%s)", variant, path, self.device, self.fp32)
        model = VideoDepthAnything(**_VARIANTS[variant])
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        self.model = model.to(self.device).eval()

    def infer(self, frames: np.ndarray, fps: float, *, input_size: int = 518) -> np.ndarray:
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"expected [T, H, W, 3] uint8 RGB, got {frames.shape}")
        # Upstream's infer_video_depth already does 32-frame windows with 10-frame overlap,
        # keyframe-based scale/shift alignment across windows and interpolation over the
        # seam. That is the temporal-consistency machinery — do not re-chunk outside it.
        with torch.inference_mode():  # upstream uses no_grad; inference_mode also skips version counters
            depths, _ = self.model.infer_video_depth(
                frames, fps, input_size=input_size, device=self.device, fp32=self.fp32
            )
        return np.asarray(depths, dtype=np.float32)


__all__ = ["VDABackend", "pick_device"]
