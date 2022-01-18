from typing import Any, Optional, Callable

import blosc as blosc
import numpy as np

from .packedarray import *

# Don't compress if array is smaller than this many bytes
_MINSIZE = 1024 ** 2 // 4


def get_pack_fn(x: Any) -> Optional[Callable]:
    if isinstance(x, np.ndarray) and x.size * x.itemsize >= _MINSIZE:
        return pack
    return None


def get_unpack_fn(x: Any) -> Optional[Callable]:
    if isinstance(x, PackedArray) and x.arraytype == ArrayType.NUMPY:
        return unpack
    return None


def pack(x: np.ndarray, **kwargs) -> PackedArray:
    return PackedArray(
        ArrayType.NUMPY,
        x.dtype,
        x.shape,
        blosc.compress_ptr(
            x.__array_interface__["data"][0],
            x.size,
            x.itemsize,
            **kwargs
        )
    )


def unpack(x: PackedArray) -> np.ndarray:
    y = np.empty(x.shape, dtype=x.dtype)
    blosc.decompress_ptr(x.buffer, y.__array_interface__['data'][0])
    return y
