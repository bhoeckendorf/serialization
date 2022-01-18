from typing import Any, Optional, Callable

import blosc as blosc
import torch

from .packedarray import *

# Don't compress if tensor is smaller than this many bytes
_MINSIZE = 1024 ** 2 // 4


def get_pack_fn(x: Any) -> Optional[Callable]:
    if isinstance(x, torch.Tensor) and (torch.as_tensor(x.shape).prod() * x.element_size()).item() >= _MINSIZE:
        return pack
    return None


def get_unpack_fn(x: Any) -> Optional[Callable]:
    if isinstance(x, PackedArray) and x.arraytype == ArrayType.PYTORCH:
        return unpack
    return None


def pack(x: torch.Tensor, **kwargs) -> PackedArray:
    x = x.detach().cpu()
    return PackedArray(
        ArrayType.PYTORCH,
        x.dtype,
        x.shape,
        blosc.compress_ptr(
            x.data_ptr(),
            torch.as_tensor(x.shape).prod().item(),
            x.element_size(),
            **kwargs
        )
    )


def unpack(x: PackedArray) -> torch.Tensor:
    y = torch.empty(x.shape, dtype=x.dtype)
    blosc.decompress_ptr(x.buffer, y.data_ptr())
    return y
