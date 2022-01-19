from typing import Any, Optional, Callable

import blosc as blosc
import numpy as np
from PIL import Image

from .packedarray import *


def get_pack_fn(x: Any) -> Optional[Callable]:
    if isinstance(x, Image.Image):
        return pack
    return None


def get_unpack_fn(x: Any) -> Optional[Callable]:
    if isinstance(x, PackedArray) and x.arraytype == ArrayType.PIL:
        return unpack
    return None


def pack(x: Image.Image, **kwargs) -> PackedArray:
    mode = x.mode
    x = np.asarray(x)
    return PackedArray(
        ArrayType.PIL,
        x.dtype,
        x.shape,
        blosc.compress_ptr(
            x.__array_interface__["data"][0],
            x.size,
            x.itemsize,
            **kwargs
        ),
        mode=mode
    )


def unpack(x: PackedArray) -> Image.Image:
    y = np.empty(x.shape, dtype=x.dtype)
    blosc.decompress_ptr(x.buffer, y.__array_interface__['data'][0])
    return Image.fromarray(y, mode=x.mode)
