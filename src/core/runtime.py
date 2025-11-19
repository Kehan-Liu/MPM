from __future__ import annotations

from typing import Any, Optional

import taichi as ti
from taichi._lib import core as _ti_core


_INITIALIZED_ARCH: Optional[Any] = None


def auto_init(arch: Optional[Any] = None, debug: bool = False) -> Any:
    """Initialize Taichi once and return the backend used."""

    global _INITIALIZED_ARCH
    if _INITIALIZED_ARCH is not None:
        return _INITIALIZED_ARCH

    backends = [
        (ti.cuda, _ti_core.with_cuda()),
        (ti.vulkan, _ti_core.with_vulkan()),
        (ti.metal, _ti_core.with_metal()),
        (ti.opengl, _ti_core.with_opengl()),
    ]
    chosen = arch
    if chosen is None:
        for candidate, available in backends:
            if available:
                chosen = candidate
                break
        else:
            chosen = ti.cpu
    ti.init(arch=chosen, default_fp=ti.f32, default_ip=ti.i32, debug=debug)
    _INITIALIZED_ARCH = chosen
    return chosen
