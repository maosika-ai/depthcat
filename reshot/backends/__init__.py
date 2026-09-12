"""Backend registry. `get_backend("vda", model="small", device="auto")`."""

from __future__ import annotations

from typing import Any

from .base import BackendInfo, DepthBackend
from .fake import FakeBackend
from .vda import VDABackend, pick_device


def get_backend(name: str, **kwargs: Any) -> DepthBackend:
    """Instantiate a backend by name. Unknown names raise `BackendError`."""
    from ..errors import BackendError

    if name == "vda":
        return VDABackend(**kwargs)
    if name == "fake":
        return FakeBackend()
    raise BackendError(f"unknown backend {name!r}")


__all__ = ["BackendInfo", "DepthBackend", "FakeBackend", "VDABackend", "get_backend", "pick_device"]
