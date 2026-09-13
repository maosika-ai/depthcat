"""Backend registry. `get_backend("vda", model="small", device="auto")` for depth,
`get_backend("dwpose", device="auto")` for pose; `"fake"` / `"fake-pose"` load nothing."""

from __future__ import annotations

from typing import Any

from .base import BackendInfo, DepthBackend
from .fake import FakeBackend
from .fake_pose import FakePoseBackend
from .vda import VDABackend, pick_device


def get_backend(name: str, **kwargs: Any) -> Any:
    """Instantiate a backend by name. Unknown names raise `BackendError`."""
    from ..errors import BackendError

    if name == "vda":
        return VDABackend(**kwargs)
    if name == "fake":
        return FakeBackend()
    if name == "dwpose":
        from ..pose.dwpose import DWPoseBackend

        return DWPoseBackend(**kwargs)
    if name == "fake-pose":
        return FakePoseBackend()
    raise BackendError(f"unknown backend {name!r}")


__all__ = ["BackendInfo", "DepthBackend", "FakeBackend", "FakePoseBackend", "VDABackend", "get_backend", "pick_device"]
