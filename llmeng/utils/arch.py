from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Literal, Tuple

if TYPE_CHECKING:
    import torch


@functools.cache
def _get_default_device_capability() -> Tuple[int, int] | None:
    import torch
    import torch.version

    if not torch.cuda.is_available() or not torch.version.cuda:
        return None
    return torch.cuda.get_device_capability()


def get_device_capability(
    device: torch.device | int | str | None = None,
) -> Tuple[int, int] | None:
    import torch
    import torch.version

    if not torch.cuda.is_available() or not torch.version.cuda:
        return None
    if device is None:
        return _get_default_device_capability()
    return torch.cuda.get_device_capability(device)


def has_device_capability(
    major: int,
    minor: int = 0,
    *,
    device: torch.device | int | str | None = None,
) -> bool:
    arch = get_device_capability(device)
    return arch == (major, minor)


def is_arch_supported(
    major: int,
    minor: int = 0,
    *,
    device: torch.device | int | str | None = None,
) -> bool:
    arch = get_device_capability(device)
    if arch is None:
        return False
    return arch >= (major, minor)


def is_sm8x(device: torch.device | int | str | None = None) -> bool:
    arch = get_device_capability(device)
    return arch is not None and arch[0] == 8


def is_sm9x(device: torch.device | int | str | None = None) -> bool:
    arch = get_device_capability(device)
    return arch is not None and arch[0] == 9


def is_sm90_supported(device: torch.device | int | str | None = None) -> bool:
    return is_arch_supported(9, 0, device=device)


def is_sm100_supported(device: torch.device | int | str | None = None) -> bool:
    return is_arch_supported(10, 0, device=device)


def get_arch_family(
    device: torch.device | int | str | None = None,
) -> Literal["ampere", "hopper", "blackwell"] | None:
    arch = get_device_capability(device)
    if arch is None or arch[0] < 8:
        return None
    if arch[0] >= 10:
        return "blackwell"
    if arch[0] >= 9:
        return "hopper"
    return "ampere"
